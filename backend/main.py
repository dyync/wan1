"""Wan2.2 TI2V-5B web API.

The API never touches the GPU. It records job state in Redis, publishes the job
to a FastStream Redis list consumed by the worker, and serves finished videos
from the shared media volume.

Redis layout (shared with worker/worker.py):
    wan:job:<task_id>   HASH   job state
    wan:pending         ZSET   queued task ids scored by submit time (queue position)
    wan:active          STRING task id the worker is currently rendering
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Path as PathParam, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faststream.redis import RedisBroker
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True)

    redis_url: str = "redis://redis:6379/0"
    media_root: Path = Path("/shared_media")
    frontend_dir: Path = Path("/app/frontend")
    job_queue: str = "wan:jobs"
    job_ttl_seconds: int = 7 * 24 * 3600
    max_pending_jobs: int = 20
    log_level: str = "INFO"


settings = Settings()
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("wan.api")

PENDING_KEY = "wan:pending"
ACTIVE_KEY = "wan:active"
TASK_ID_PATTERN = r"^[0-9a-f]{32}$"

JobStatus = Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
VideoSize = Literal["1280*704", "704*1280"]


def job_key(task_id: str) -> str:
    return f"wan:job:{task_id}"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=3, max_length=2000)
    size: VideoSize = "1280*704"
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)

    @field_validator("prompt")
    @classmethod
    def normalise_prompt(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("prompt must contain at least 3 non-whitespace characters")
        return value


class GenerationJob(BaseModel):
    """Message published to the worker queue. Mirrored in worker/worker.py."""

    task_id: str
    prompt: str
    size: VideoSize
    seed: int | None = None


class TaskAccepted(BaseModel):
    task_id: str
    status: JobStatus
    status_url: str


class TaskStatus(BaseModel):
    task_id: str
    status: JobStatus
    stage: str | None = None
    progress: float = Field(ge=0, le=100)
    queue_position: int | None = None
    prompt: str
    size: str
    seed: int | None = None
    file_path: str | None = None
    video_url: str | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class Health(BaseModel):
    status: Literal["ok"]
    redis: bool
    worker_busy: bool
    pending_jobs: int


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

broker = RedisBroker(settings.redis_url)
redis: aioredis.Redis


@asynccontextmanager
async def lifespan(_: FastAPI):
    global redis
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()
    await broker.connect()
    log.info("connected to %s", settings.redis_url)
    try:
        yield
    finally:
        await broker.close()
        await redis.aclose()


app = FastAPI(title="Wan2.2 TI2V-5B", version="1.0.0", lifespan=lifespan)

# Only the finished-videos directory is exposed; in-progress renders live elsewhere.
# check_dir=False: the worker creates the directory on first start.
app.mount(
    "/media/videos",
    StaticFiles(directory=settings.media_root / "videos", check_dir=False),
    name="videos",
)


def _opt_float(value: str | None) -> float | None:
    return float(value) if value else None


def _opt_int(value: str | None) -> int | None:
    return int(value) if value else None


@app.post("/api/tasks", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_task(body: TaskCreate) -> TaskAccepted:
    pending = await redis.zcard(PENDING_KEY)
    if pending >= settings.max_pending_jobs:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Queue is full ({pending} jobs waiting). Try again later.",
        )

    task_id = uuid.uuid4().hex
    now = time.time()
    key = job_key(task_id)

    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(
            key,
            mapping={
                "task_id": task_id,
                "status": "PENDING",
                "stage": "queued",
                "progress": "0",
                "prompt": body.prompt,
                "size": body.size,
                "seed": "" if body.seed is None else str(body.seed),
                "created_at": f"{now:.3f}",
            },
        )
        pipe.expire(key, settings.job_ttl_seconds)
        pipe.zadd(PENDING_KEY, {task_id: now})
        await pipe.execute()

    try:
        await broker.publish(
            GenerationJob(task_id=task_id, prompt=body.prompt, size=body.size, seed=body.seed),
            list=settings.job_queue,
        )
    except Exception as exc:
        log.exception("failed to enqueue %s", task_id)
        async with redis.pipeline(transaction=True) as pipe:
            pipe.zrem(PENDING_KEY, task_id)
            pipe.hset(key, mapping={"status": "FAILED", "stage": "failed", "error": "Could not enqueue job."})
            await pipe.execute()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Task queue unavailable.") from exc

    log.info("queued %s (%s)", task_id, body.size)
    return TaskAccepted(task_id=task_id, status="PENDING", status_url=f"/api/tasks/{task_id}")


@app.get("/api/tasks/{task_id}", response_model=TaskStatus)
async def get_task(task_id: Annotated[str, PathParam(pattern=TASK_ID_PATTERN)]) -> TaskStatus:
    async with redis.pipeline(transaction=False) as pipe:
        pipe.hgetall(job_key(task_id))
        pipe.zrank(PENDING_KEY, task_id)
        data, rank = await pipe.execute()

    if not data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown or expired task.")

    job_status: JobStatus = data["status"]
    filename = data.get("filename")
    completed = job_status == "COMPLETED" and filename

    return TaskStatus(
        task_id=task_id,
        status=job_status,
        stage=data.get("stage"),
        progress=float(data.get("progress") or 0),
        queue_position=(rank + 1) if job_status == "PENDING" and rank is not None else None,
        prompt=data.get("prompt", ""),
        size=data.get("size", ""),
        seed=_opt_int(data.get("seed")),
        file_path=str(settings.media_root / "videos" / filename) if completed else None,
        video_url=f"/media/videos/{filename}" if completed else None,
        error=data.get("error") or None,
        created_at=float(data["created_at"]),
        started_at=_opt_float(data.get("started_at")),
        finished_at=_opt_float(data.get("finished_at")),
    )


@app.get("/healthz", response_model=Health)
async def healthz() -> Health:
    try:
        async with redis.pipeline(transaction=False) as pipe:
            pipe.ping()
            pipe.exists(ACTIVE_KEY)
            pipe.zcard(PENDING_KEY)
            _, active, pending = await pipe.execute()
    except aioredis.RedisError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable.") from exc
    return Health(status="ok", redis=True, worker_busy=bool(active), pending_jobs=pending)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html", headers={"Cache-Control": "no-cache"})
