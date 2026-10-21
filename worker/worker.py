"""Wan2.2 TI2V-5B GPU worker.

Consumes jobs from a FastStream Redis list one at a time and runs the official
Wan2.2 `generate.py` as a subprocess:

    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir <ckpt> \
        --offload_model True --convert_model_dtype --t5_cpu \
        --prompt <prompt> --base_seed <seed> --save_file <partial.mp4>

Each job gets a fresh process, so all VRAM is released between jobs, and a CUDA
OOM or crash fails one job without taking the worker down. The prompt is passed
as an argv element (no shell), so it cannot inject commands.

Upstream generate.py has no `--output_dir` flag. It writes to `--save_file`.
The worker renders into /shared_media/.partial and atomically moves the finished
file into /shared_media/videos, so the API never serves a half-written video.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import shutil
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Literal

import imageio_ffmpeg
import redis.asyncio as aioredis
from faststream import FastStream
from faststream.redis import RedisBroker
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True)

    redis_url: str = "redis://redis:6379/0"
    job_queue: str = "wan:jobs"
    media_root: Path = Path("/shared_media")
    wan_repo_dir: Path = Path("/opt/Wan2.2")
    ckpt_dir: Path = Path("/models/Wan2.2-TI2V-5B")
    hf_repo_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    job_timeout_seconds: int = 3600
    sample_steps: int | None = None
    log_level: str = "INFO"


settings = Settings()
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("wan.worker")

PENDING_KEY = "wan:pending"
ACTIVE_KEY = "wan:active"
VIDEOS_DIR = settings.media_root / "videos"
PARTIAL_DIR = settings.media_root / ".partial"
# Files that must exist for the checkpoint to be usable (names from wan/configs/wan_ti2v_5B.py).
REQUIRED_CKPT_FILES = ("config.json", "Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth")
DOWNLOAD_MARKER = ".download-complete"


def job_key(task_id: str) -> str:
    return f"wan:job:{task_id}"


class GenerationJob(BaseModel):
    """Mirrors backend/main.py GenerationJob."""

    task_id: str
    prompt: str
    size: Literal["1280*704", "704*1280"]
    seed: int | None = None


broker = RedisBroker(settings.redis_url)
app = FastStream(broker)
redis: aioredis.Redis


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Turns generate.py's log/tqdm output into a stage and a percentage stored in Redis.

    Progress bands: loading 0-5, sampling 5-90 (per denoising step),
    VAE decode 90-97, saving/finalising 97-100.
    """

    TQDM_RE = re.compile(r"(\d+)/(\d+) \[")

    def __init__(self, key: str) -> None:
        self.key = key
        self.stage = "loading_model"
        self.progress = 1.0
        self.tail: deque[str] = deque(maxlen=30)
        self._seen_generate = False
        self._last_write = 0.0
        self._written: tuple[str, float] | None = None

    async def feed(self, line: str) -> None:
        match = self.TQDM_RE.search(line)
        if not match:
            self.tail.append(line)
            log.info("[wan] %s", line)

        if "Generating video" in line:
            self._seen_generate = True
            await self._set("sampling", 5.0)
        elif "Saving generated video" in line:
            await self._set("saving", 97.0)
        elif match:
            step, total = int(match.group(1)), int(match.group(2))
            # Ignore unrelated bars (e.g. checkpoint shard loading) before sampling starts.
            if total and (self._seen_generate or total >= 20) and self.stage in ("loading_model", "sampling"):
                if step >= total:
                    log.info("[wan] sampling finished (%d steps), decoding", total)
                    await self._set("decoding", 90.0)
                else:
                    if step % 10 == 0:
                        log.info("[wan] sampling step %d/%d", step, total)
                    await self._set("sampling", 5.0 + 85.0 * step / total)

    async def _set(self, stage: str, progress: float, force: bool = False) -> None:
        self.stage, self.progress = stage, round(progress, 1)
        now = time.monotonic()
        state = (self.stage, self.progress)
        stage_changed = self._written is None or self._written[0] != stage
        if state == self._written or (not force and not stage_changed and now - self._last_write < 1.0):
            return
        await redis.hset(self.key, mapping={"stage": stage, "progress": str(self.progress)})
        self._written, self._last_write = state, now

    async def set(self, stage: str, progress: float) -> None:
        await self._set(stage, progress, force=True)


async def _pump_output(proc: asyncio.subprocess.Process, tracker: ProgressTracker) -> None:
    """Read merged stdout/stderr, splitting on both \\n and tqdm's \\r."""
    assert proc.stdout is not None
    buffer = b""
    while chunk := await proc.stdout.read(4096):
        buffer += chunk
        *lines, buffer = re.split(rb"[\r\n]", buffer)
        for raw in lines:
            if line := raw.decode("utf-8", errors="replace").strip():
                await tracker.feed(line)
    if line := buffer.decode("utf-8", errors="replace").strip():
        await tracker.feed(line)


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=15)
    except TimeoutError:
        proc.kill()
        await proc.wait()


async def _faststart(src: Path, dst: Path) -> None:
    """Move the MP4 moov atom to the front so browsers can start playback before the whole file loads."""
    tmp = dst.with_suffix(".faststart.mp4")
    proc = await asyncio.create_subprocess_exec(
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(tmp),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        os.replace(tmp, dst)
        src.unlink(missing_ok=True)
    else:
        log.warning("faststart remux failed (%s); publishing original file", err.decode(errors="replace").strip())
        tmp.unlink(missing_ok=True)
        os.replace(src, dst)
    dst.chmod(0o644)


async def _fail(key: str, message: str) -> None:
    await redis.hset(
        key,
        mapping={"status": "FAILED", "stage": "failed", "error": message[:4000], "finished_at": f"{time.time():.3f}"},
    )


# ---------------------------------------------------------------------------
# Job handler
# ---------------------------------------------------------------------------


@broker.subscriber(list=settings.job_queue)
async def handle_job(job: GenerationJob) -> None:
    key = job_key(job.task_id)
    if not await redis.exists(key):
        log.warning("job %s has no state (expired?); skipping", job.task_id)
        await redis.zrem(PENDING_KEY, job.task_id)
        return

    seed = job.seed if job.seed is not None else random.randint(0, 2**31 - 1)
    partial = PARTIAL_DIR / f"{job.task_id}.mp4"
    final = VIDEOS_DIR / f"{job.task_id}.mp4"
    started = time.time()

    async with redis.pipeline(transaction=True) as pipe:
        pipe.zrem(PENDING_KEY, job.task_id)
        pipe.set(ACTIVE_KEY, job.task_id)
        pipe.hset(
            key,
            mapping={
                "status": "PROCESSING",
                "stage": "loading_model",
                "progress": "1",
                "seed": str(seed),
                "started_at": f"{started:.3f}",
            },
        )
        await pipe.execute()

    cmd = [
        sys.executable, "generate.py",
        "--task", "ti2v-5B",
        "--size", job.size,
        "--ckpt_dir", str(settings.ckpt_dir),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--base_seed", str(seed),
        "--prompt", job.prompt,
        "--save_file", str(partial),
    ]
    if settings.sample_steps:
        cmd += ["--sample_steps", str(settings.sample_steps)]

    log.info("job %s: start size=%s seed=%d prompt=%r", job.task_id, job.size, seed, job.prompt[:120])
    tracker = ProgressTracker(key)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=settings.wan_repo_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        await asyncio.wait_for(_pump_output(proc, tracker), timeout=settings.job_timeout_seconds)
        returncode = await proc.wait()
    except TimeoutError:
        await _terminate(proc)
        partial.unlink(missing_ok=True)
        await _fail(key, f"Generation exceeded the {settings.job_timeout_seconds}s time limit.")
        log.error("job %s: timed out", job.task_id)
        await redis.delete(ACTIVE_KEY)
        return
    except asyncio.CancelledError:
        # Worker shutting down mid-job: don't leave an orphan process holding the GPU.
        await _terminate(proc)
        partial.unlink(missing_ok=True)
        await _fail(key, "Worker was restarted during generation. Please resubmit.")
        await redis.delete(ACTIVE_KEY)
        raise

    try:
        if returncode != 0 or not partial.exists() or partial.stat().st_size == 0:
            tail = "\n".join(tracker.tail)
            hint = " (CUDA out of memory)" if "out of memory" in tail.lower() else ""
            await _fail(key, f"generate.py exited with code {returncode}{hint}.\n\n{tail}")
            partial.unlink(missing_ok=True)
            log.error("job %s: failed with code %s", job.task_id, returncode)
            return

        await tracker.set("finalizing", 98.0)
        await _faststart(partial, final)

        finished = time.time()
        await redis.hset(
            key,
            mapping={
                "status": "COMPLETED",
                "stage": "done",
                "progress": "100",
                "filename": final.name,
                "finished_at": f"{finished:.3f}",
            },
        )
        log.info("job %s: completed in %.0fs -> %s", job.task_id, finished - started, final)
    except Exception as exc:
        log.exception("job %s: post-processing failed", job.task_id)
        partial.unlink(missing_ok=True)
        await _fail(key, f"Post-processing failed: {exc}")
    finally:
        await redis.delete(ACTIVE_KEY)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def ensure_weights() -> None:
    ckpt = settings.ckpt_dir
    marker = ckpt / DOWNLOAD_MARKER
    if marker.exists() and all((ckpt / name).exists() for name in REQUIRED_CKPT_FILES):
        log.info("weights present at %s", ckpt)
        return

    ckpt.mkdir(parents=True, exist_ok=True)
    cli = shutil.which("hf")
    cmd = (
        [cli, "download", settings.hf_repo_id, "--local-dir", str(ckpt)]
        if cli
        else ["huggingface-cli", "download", settings.hf_repo_id, "--local-dir", str(ckpt)]
    )
    log.info("downloading %s to %s (first start only, ~30 GB)", settings.hf_repo_id, ckpt)
    # Retry: large downloads on rented instances drop now and then; the CLI resumes partial files.
    for attempt in range(1, 6):
        proc = await asyncio.create_subprocess_exec(*cmd)
        if await proc.wait() == 0:
            break
        log.warning("download attempt %d failed (exit %s); retrying", attempt, proc.returncode)
        await asyncio.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"failed to download {settings.hf_repo_id}")

    missing = [name for name in REQUIRED_CKPT_FILES if not (ckpt / name).exists()]
    if missing:
        raise RuntimeError(f"checkpoint at {ckpt} is incomplete, missing: {missing}")
    marker.write_text(f"{time.time():.0f}\n")
    log.info("weights ready")


@app.on_startup
async def startup() -> None:
    global redis
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()

    for directory in (VIDEOS_DIR, PARTIAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o755)
    for leftover in PARTIAL_DIR.glob("*"):
        leftover.unlink(missing_ok=True)

    # A job left in "active" means the previous worker died (e.g. host OOM kill) mid-render.
    if stale := await redis.get(ACTIVE_KEY):
        log.warning("marking interrupted job %s as failed", stale)
        if await redis.exists(job_key(stale)):
            await _fail(job_key(stale), "Worker crashed during generation. Please resubmit.")
        await redis.delete(ACTIVE_KEY)

    # Jobs keep queueing in Redis while weights download; consumption starts afterwards.
    await ensure_weights()
    commit_file = settings.wan_repo_dir / ".wan-commit"
    commit = commit_file.read_text().strip() if commit_file.exists() else "unknown"
    log.info("worker ready (Wan2.2 %s), consuming %s", commit[:12], settings.job_queue)


@app.on_shutdown
async def shutdown() -> None:
    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(app.run())
