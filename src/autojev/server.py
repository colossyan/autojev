"""Serve the trained checkpoint through the TypeSafe decision API."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import os
import queue
import threading
import time
import uuid
from _thread import LockType
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint

from autojev.types import Answer, DecisionInput, DecisionResponse, JSONValue, Question as DecisionQuestion

if TYPE_CHECKING:
    from autojev.model import DecisionModel

type Content = str | dict[str, JsonValue] | list[JsonValue]
DEFAULT_MODEL = "autojev-qwen3.8-27b"
ALIASES = {"autojev", "jev-latest", "jev-preview", "jev-1.13.0", DEFAULT_MODEL}


# Rows per forward pass, as before; what changes is that a pass is filled from
# every request waiting, not from one request while the rest are turned away.
BATCH_ROWS = int(os.getenv("AUTOJEV_BATCH_ROWS", "8"))
# Padded tokens one pass may hold: rows times the longest of them. A pass of
# eight short questions and a pass of eight 7k-token plans are not the same
# memory — the second ran the card out of memory mid-suite.
BATCH_TOKENS = int(os.getenv("AUTOJEV_BATCH_TOKENS", "24576"))
# Rows that may wait for a pass before a request is told the model is busy.
MAX_WAITING_ROWS = int(os.getenv("AUTOJEV_MAX_WAITING_ROWS", "2048"))
# Fewer questions than this are read whole, beside other requests' rows: a
# shared read is two passes (the state, then the questions), each paying the
# fixed cost of a forward, and a request of two or three questions saved
# less than that pass cost.
SHARED_MIN_ROWS = int(os.getenv("AUTOJEV_SHARED_MIN_ROWS", "5"))
# Shared requests waiting together are read together, up to this many
# requests and questions: their questions fill common batches.
SHARED_JOBS = int(os.getenv("AUTOJEV_SHARED_JOBS", "4"))
SHARED_ROWS = int(os.getenv("AUTOJEV_SHARED_ROWS", "96"))
# The whole process's GPU budget. The card is shared: within it the model
# holds what it needs and gives none of it back, beyond it nothing. Measured
# on Palma's GPU3: the old ceiling-less server peaked at 80 GB, and memory it
# released after a batch was taken by a neighbour's training, after which
# every request ran out (396 of 400 in one hour).
GPU_CAP = int(float(os.getenv("AUTOJEV_GPU_CAP_GB", "70")) * 2**30)
# What the process holds outside PyTorch's allocator (the CUDA context,
# library workspaces), which the per-process fraction does not count.
_CONTEXT_MARGIN = int(1.5 * 2**30)


def claim_gpu_budget(torch) -> None:
    """Cap PyTorch at the budget and claim all of it now, while it is free:
    an allocation past the cap fails here (and the batch is split) rather
    than taking more of the card, and freed blocks stay with the process."""
    if not GPU_CAP or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    budget = max(GPU_CAP - _CONTEXT_MARGIN, 0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total))
    spare = budget - torch.cuda.memory_reserved() - 2**28
    if spare > 0:
        try:
            block = torch.empty(spare, dtype=torch.uint8, device="cuda")
            del block       # back to the allocator's cache, not to the card
        except torch.cuda.OutOfMemoryError:
            print("autojev: could not claim the whole GPU budget up front; "
                  "it is taken as requests need it", flush=True)
    print(f"autojev: GPU budget {GPU_CAP / 2**30:.0f} GiB; holding "
          f"{torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)


@dataclass
class Service:
    model: DecisionModel | None = None
    name: str = DEFAULT_MODEL
    checkpoint: str = "checkpoints/selected"
    release_date: str = ""
    lock: LockType = field(default_factory=threading.Lock)
    # Whether a request's questions read their shared state once; on only
    # when it answered a sample exactly as the full read does.
    shared: bool = False
    cache: Any = None


service = Service()


@dataclass
class _Pending:
    row: DecisionInput
    question: DecisionQuestion
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    tokens: int = 0


@dataclass
class _Shared:
    """A request whose questions share their state, answered as one job."""
    rows: list[DecisionInput]
    questions: list[DecisionQuestion]
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


def _estimate_tokens(row: DecisionInput) -> int:
    """Roughly the row's length in tokens, without tokenizing it: under three
    characters a token on this text, so this errs long."""
    import json

    return len(json.dumps([row["state"], row["question"]], ensure_ascii=False)) * 10 // 26 + 256


class Batcher:
    """One worker thread that answers questions from every request at once.

    The server used to take one request at a time and answer the rest 529,
    so a caller asking a single question waited out whoever was asking, and
    a pass ran with as few rows as that one request had. Each request now
    queues its questions and the worker fills every pass, up to BATCH_ROWS,
    from whatever is waiting — the same pass size, so the same memory."""

    def __init__(self) -> None:
        self._queue: queue.Queue[_Pending] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="autojev-batcher", daemon=True)
            self._thread.start()

    def waiting(self) -> int:
        return self._queue.qsize()

    async def answer(self, rows: list[DecisionInput], questions: list[DecisionQuestion],
                     cached: bool = True) -> list[tuple[Answer, int]]:
        """Each row's answer: given again when the same row was answered
        before (``answer_cache``), read by the model otherwise. A caller
        measuring the model itself asks with ``cached`` off."""
        cache = service.cache if cached else None
        if cache is None:
            return await self._uncached(rows, questions)
        keys = [cache.key(row, question) for row, question in zip(rows, questions, strict=True)]
        known = cache.get_many(keys)
        out: list[tuple[Answer, int] | None] = [(known[k], 0) if k in known else None for k in keys]
        todo = [i for i, k in enumerate(keys) if k not in known]
        if todo:
            read = await self._uncached([rows[i] for i in todo], [questions[i] for i in todo])
            for i, result in zip(todo, read, strict=True):
                out[i] = result
            cache.put_many([(keys[i], result[0]) for i, result in zip(todo, read, strict=True)])
        return [r for r in out if r is not None]

    async def _uncached(self, rows: list[DecisionInput], questions: list[DecisionQuestion]) -> list[tuple[Answer, int]]:
        loop = asyncio.get_running_loop()
        if service.shared and len(rows) >= max(SHARED_MIN_ROWS, 2) and not any(row.get("images") for row in rows):
            job = _Shared(rows, questions, loop, loop.create_future())
            self._queue.put(job)  # type: ignore[arg-type]
            return await job.future
        pending = [_Pending(row, question, loop, loop.create_future(), _estimate_tokens(row))
                   for row, question in zip(rows, questions, strict=True)]
        for item in pending:
            self._queue.put(item)
        return list(await asyncio.gather(*(item.future for item in pending)))

    def _run(self) -> None:
        import torch
        from autojev.model import answer

        held: _Pending | None = None
        while True:
            first = held or self._queue.get()
            held = None
            if isinstance(first, _Shared):
                jobs = [first]
                rows = len(first.rows)
                # The shared requests queued right behind it are read with it.
                with self._queue.mutex:
                    waiting = self._queue.queue
                    while (waiting and isinstance(waiting[0], _Shared) and len(jobs) < SHARED_JOBS
                           and rows + len(waiting[0].rows) <= SHARED_ROWS):
                        job = waiting.popleft()
                        jobs.append(job)
                        rows += len(job.rows)
                self._shared(jobs, answer, torch)
                continue
            items = [first]
            longest = items[0].tokens
            while len(items) < BATCH_ROWS:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, _Shared) or (len(items) + 1) * max(longest, item.tokens) > BATCH_TOKENS:
                    held = item     # first in the next pass, so it keeps its turn
                    break
                items.append(item)
                longest = max(longest, item.tokens)
            self._answer(items, answer, torch)

    @staticmethod
    def _shared(jobs: list[_Shared], answer, torch) -> None:
        try:
            model = service.model
            if model is None:
                raise RuntimeError("The model is not ready.")
            try:
                read_all = _profiled(lambda: model.shared_distributions_many([job.rows for job in jobs]), torch)
            except torch.cuda.OutOfMemoryError:
                # Read together they did not fit in the budget; one at a time
                # they do, and nobody gets an error.
                if len(jobs) == 1:
                    raise
                read_all = [model.shared_distributions(job.rows) for job in jobs]
        except Exception as error:  # noqa: BLE001
            for job in jobs:
                job.loop.call_soon_threadsafe(_settle, job.future, None, error)
            return
        for job, (distributions, read) in zip(jobs, read_all, strict=True):
            try:
                share = read // max(len(job.rows), 1)
                result = [(answer(q, values[:len(_options(q))]), share)
                          for q, values in zip(job.questions, distributions, strict=True)]
            except Exception as error:  # noqa: BLE001
                job.loop.call_soon_threadsafe(_settle, job.future, None, error)
                continue
            job.loop.call_soon_threadsafe(_settle, job.future, result, None)

    def _answer(self, items: list[_Pending], answer, torch) -> None:
        try:
            self._pass(items, answer, torch)
        except torch.cuda.OutOfMemoryError as error:
            # Too much for one pass after all: halve it, down to one row,
            # rather than fail every caller who shared it.
            if len(items) == 1:
                items[0].loop.call_soon_threadsafe(_settle, items[0].future, None, error)
                return
            half = len(items) // 2
            self._answer(items[:half], answer, torch)
            self._answer(items[half:], answer, torch)
        except Exception:  # noqa: BLE001 — find whose row it was
            # One caller's bad question must not fail the others it shared
            # the pass with: answer each on its own, so only the owner of
            # the bad row hears the error.
            for item in items:
                try:
                    self._pass([item], answer, torch)
                except Exception as error:  # noqa: BLE001
                    item.loop.call_soon_threadsafe(_settle, item.future, None, error)

    @staticmethod
    def _pass(items: list[_Pending], answer, torch) -> None:
        model = service.model
        if model is None:
            raise RuntimeError("The model is not ready.")
        import os
        import time as _time
        started = _time.perf_counter()
        with torch.inference_mode():
            batch = model.prepare([item.row for item in items])
            distributions = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
        if os.getenv("AUTOJEV_TIMING") == "1":
            images = sum(len(item.row.get("images") or ()) for item in items)
            print(f"autojev timing: pass {len(items)} rows {batch.input_tokens}t "
                  f"{images} images={(_time.perf_counter() - started) * 1000:.0f}ms", flush=True)
        share = batch.input_tokens // max(len(items), 1)
        results = [(answer(item.question, values[:count]), share)
                   for item, values, count in zip(items, distributions, batch.counts, strict=True)]
        for item, result in zip(items, results, strict=True):
            item.loop.call_soon_threadsafe(_settle, item.future, result, None)


_profiled_seen = [0]


def _profiled(run, torch):
    """``run()``; with ``AUTOJEV_PROFILE_DIR`` set, requests 20 to 29 are
    read under the torch profiler and its tables written there."""
    import os

    out = os.getenv("AUTOJEV_PROFILE_DIR")
    _profiled_seen[0] += 1
    if not out or not 20 <= _profiled_seen[0] < 30:
        return run()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = run()
        torch.cuda.synchronize()
    n = _profiled_seen[0]
    with open(os.path.join(out, f"profile_{n}.txt"), "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        f.write("\n\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))
    return result


def _options(question: DecisionQuestion) -> list:
    from autojev.model import options

    return options(question)[0]


def _shared_matches(model: DecisionModel) -> bool:
    """Whether reading the shared state once answers as reading every row
    whole does, on a request shaped like the ones served: one long state,
    questions of each type."""
    import torch

    state = ("A scene in a product video. " * 120 + "It shows a presenter in a bright "
             "office who says one line to camera, then a slow push-in on a dashboard.")
    questions: list[DecisionQuestion] = [
        {"type": "noul", "instructions": "Does the scene have speech?"},
        {"type": "choice", "instructions": "Which canvas?",
         "criteria": {"wide": "16:9", "tall": "9:16", "square": "1:1"}},
        {"type": "noul", "instructions": "Is there background music?"},
        # Two of one kind, opening on the same rules: read through a group's
        # own longer prefix, which has to answer as the whole read too.
        *({"type": "choice", "instructions": "Below is every scene in one video. " * 40
           + f"Which output does scene {n} take from the first scene?",
           "criteria": {"frame": "its start image", "audio": "its narration", "none": "nothing"}}
          for n in (2, 3)),
    ]
    worst = 0.0
    jobs: list = []
    # Read cold; then read on from what the first kept (a longer state that
    # opens the same way); then the same again, wholly from what was kept.
    for told in (state, state + " The dashboard shows revenue by quarter.",
                 state + " The dashboard shows revenue by quarter."):
        rows: list[DecisionInput] = [{"state": told, "question": q, "images": []} for q in questions]
        with torch.inference_mode():
            batch = model.prepare(rows)
            whole = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
        shared, _ = model.shared_distributions(rows)
        worst = max(worst, max(abs(a - b) for w, sh, c in zip(whole, shared, batch.counts)
                               for a, b in zip(w[:c], sh[:c])))
        jobs.append((rows, whole, batch.counts))
    # Read together, the requests' questions share batches over a cache
    # merged from states of different lengths: the same answers again.
    # From nothing kept, so the states of different lengths are read in one
    # left-padded pass as well.
    model.__dict__.get("_kept", {}).clear()
    together = model.shared_distributions_many([rows for rows, _, _ in jobs])
    for (_, whole, counts), (shared, _) in zip(jobs, together, strict=True):
        worst = max(worst, max(abs(a - b) for w, sh, c in zip(whole, shared, counts)
                               for a, b in zip(w[:c], sh[:c])))
    print(f"autojev: shared-state reading differs by at most {worst:.4f} from the whole read", flush=True)
    return worst < 0.02


def _settle(future: asyncio.Future, result, error: BaseException | None) -> None:
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(result)


batcher = Batcher()


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: Content | None = None


class Choice(Question):
    type: Literal["choice"]
    criteria: dict[str, Content | None] = Field(min_length=1, max_length=255)


class Score(Question):
    type: Literal["score"]
    criteria: list[Content] = Field(min_length=2, max_length=10)


class Noul(Question):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], Content | None] | None = None


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    state: Content
    questions: dict[str, Annotated[Choice | Score | Noul, Field(discriminator="type")]] = Field(min_length=1)
    images: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str) -> str:
        if value not in ALIASES | {service.name}:
            raise ValueError(f"Unknown model. Use {service.name} or jev-latest.")
        return value

    @field_validator("images")
    @classmethod
    def valid_images(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value) > 12_000_000:
                raise ValueError("Each image must be at most 8 MB before base64 encoding.")
            header, separator, encoded = value.partition(",")
            if not separator or header not in {
                "data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64",
            }:
                raise ValueError("Images must be base64 PNG, JPEG, or WebP data URLs.")
            try:
                content = base64.b64decode(encoded, validate=True)
                if len(content) > 8_000_000:
                    raise ValueError("Each image must be at most 8 MB.")
                with Image.open(BytesIO(content)) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError("Each image must have at most 16 million pixels.")
                    if image.format not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError("Unsupported image format.")
                    image.verify()
            except (binascii.Error, OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                raise ValueError("Invalid image data.") from error
        return values


def authenticate(authorization: str | None = Header(default=None)) -> None:
    key = os.getenv("AUTOJEV_API_KEY")
    if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "Missing or invalid API key.", headers={"WWW-Authenticate": "Bearer"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from autojev.model import DecisionModel

    service.checkpoint = os.getenv("AUTOJEV_CHECKPOINT", "checkpoints/selected")
    service.model = await run_in_threadpool(DecisionModel, checkpoint=service.checkpoint)
    import torch
    claim_gpu_budget(torch)
    from autojev.answer_cache import open_cache
    service.cache = open_cache(service.checkpoint)
    service.name = f"autojev-{service.model.base_model.rsplit('/', 1)[-1].lower()}"
    modified = (Path(service.checkpoint) / "decision_config.json").stat().st_mtime
    service.release_date = datetime.fromtimestamp(modified, timezone.utc).date().isoformat()
    if os.getenv("AUTOJEV_SHARED_STATE", "1") != "0":
        try:
            service.shared = await run_in_threadpool(_shared_matches, service.model)
        except Exception as error:  # noqa: BLE001 — the whole read still serves
            print(f"autojev: shared-state reading unavailable ({error})", flush=True)
            service.shared = False
    batcher.start()
    try:
        yield
    finally:
        service.model = None


app = FastAPI(title="AutoJev", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def request_metadata(request: Request, call_next: RequestResponseEndpoint) -> Response:
    started, identifier = time.perf_counter(), uuid.uuid4().hex
    response = await call_next(request)
    response.headers["x-typesafe-request-id"] = identifier
    response.headers["x-request-id"] = identifier
    response.headers["server-timing"] = f"total;dur={(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def playground() -> str:
    return Path(__file__).with_name("playground.html").read_text()


@app.get("/health", response_model=None)
def health() -> dict[str, JSONValue]:
    return {"status": "ready" if service.model is not None else "loading", "model": service.name,
            "checkpoint": service.checkpoint, "authentication": bool(os.getenv("AUTOJEV_API_KEY")),
            "modalities": ["text", "image"], "shared_state": service.shared}


@app.get("/v1/models", dependencies=[Depends(authenticate)], response_model=None)
def models() -> dict[str, JSONValue]:
    return {"models": [
        {"name": name, "description": "Local AutoJev text and image decisions.", "release_date": service.release_date}
        for name in sorted(ALIASES | {service.name})
    ]}


def predict(model: DecisionModel, body: EvaluationRequest) -> DecisionResponse:
    import torch
    from autojev.model import answer

    questions = {key: cast(DecisionQuestion, question.model_dump(exclude_none=True))
                 for key, question in body.questions.items()}
    identifiers = list(questions)
    rows: list[DecisionInput] = [{"state": body.state, "question": question, "images": list(body.images)}
                                 for question in questions.values()]
    answers: dict[str, Answer] = {}
    input_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = model.prepare(rows[start:start + 8])
            distributions: list[list[float]] = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
            for identifier, values, count in zip(identifiers[start:start + 8], distributions, batch.counts, strict=True):
                answers[identifier] = answer(questions[identifier], values[:count])
            input_tokens += batch.input_tokens
    return {"model": service.name, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


@app.post("/v1/systemone", dependencies=[Depends(authenticate)], response_model=None)
async def system_one(body: EvaluationRequest,
                     x_autojev_cache: str | None = Header(default=None)) -> DecisionResponse:
    if service.model is None:
        raise HTTPException(503, "The model is not ready.")
    questions = {key: cast(DecisionQuestion, question.model_dump(exclude_none=True))
                 for key, question in body.questions.items()}
    if batcher.waiting() + len(questions) > MAX_WAITING_ROWS:
        raise HTTPException(529, "The model is busy. Retry shortly.", headers={"Retry-After": "1"})
    rows: list[DecisionInput] = [{"state": body.state, "question": question, "images": list(body.images)}
                                 for question in questions.values()]
    try:
        results = await batcher.answer(rows, list(questions.values()),
                                       cached=(x_autojev_cache or "").lower() != "off")
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    answers: dict[str, Answer] = {key: result for key, (result, _) in zip(questions, results, strict=True)}
    input_tokens = sum(tokens for _, tokens in results)
    return {"model": service.name, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def main() -> None:
    import uvicorn

    uvicorn.run("autojev.server:app", host=os.getenv("AUTOJEV_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
