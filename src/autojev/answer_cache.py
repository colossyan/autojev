"""Answers already given, kept on disk and given again.

A decision is a function of the checkpoint, the code that builds its prompt,
and the row itself (state, question, images): asked the same row again, the
model reads the same tokens and answers the same distribution. Measured on
two hybrid CI runs of one commit, 79% of their 2,055 rows were asked
identically by both — read again, they were most of the run's GPU time on a
model shared with every other tier.

The key covers everything the answer depends on, so a new checkpoint or a
change to how prompts are built asks everything afresh. ``AUTOJEV_ANSWER_CACHE``
names the file (``off`` disables it); ``AUTOJEV_ANSWER_CACHE_ROWS`` bounds it,
oldest-used first out. A request with ``X-AutoJev-Cache: off`` is read by the model
whatever is kept, and keeps nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_DEFAULT = Path.home() / ".cache" / "autojev" / "answers.sqlite"
_MAX_ROWS = int(os.getenv("AUTOJEV_ANSWER_CACHE_ROWS", "500000"))


class AnswerCache:
    def __init__(self, path: Path, fingerprint: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS answers "
                         "(key TEXT PRIMARY KEY, value TEXT NOT NULL, used REAL NOT NULL)")
        self._lock = threading.Lock()
        self._fingerprint = fingerprint
        self.hits = 0
        self.misses = 0

    def key(self, row: dict[str, Any], question: dict[str, Any]) -> str:
        blob = json.dumps([self._fingerprint, row.get("state"), question, row.get("images") or []],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, Any]:
        if not keys:
            return {}
        with self._lock:
            found = dict(self._db.execute(
                f"SELECT key, value FROM answers WHERE key IN ({','.join('?' * len(keys))})",
                list(keys)).fetchall())
            if found:
                self._db.executemany("UPDATE answers SET used=? WHERE key=?",
                                     [(time.time(), k) for k in found])
        self.hits += len(found)
        self.misses += len(set(keys) - set(found))
        return {k: json.loads(v) for k, v in found.items()}

    def put_many(self, items: Sequence[tuple[str, Any]]) -> None:
        if not items:
            return
        now = time.time()
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO answers (key, value, used) VALUES (?, ?, ?)",
                                 [(k, json.dumps(v), now) for k, v in items])
            (count,) = self._db.execute("SELECT COUNT(*) FROM answers").fetchone()
            if count > _MAX_ROWS:
                self._db.execute("DELETE FROM answers WHERE key IN (SELECT key FROM answers "
                                 "ORDER BY used LIMIT ?)", (count - _MAX_ROWS + _MAX_ROWS // 10,))


def fingerprint(checkpoint: str | Path) -> str:
    """What every answer depends on besides its row: the checkpoint's own
    config (its weights' provenance and temperature) and the code that turns
    a row into the model's prompt."""
    h = hashlib.sha256()
    h.update((Path(checkpoint) / "decision_config.json").read_bytes())
    h.update((Path(__file__).parent / "model.py").read_bytes())
    return h.hexdigest()


def open_cache(checkpoint: str | Path) -> AnswerCache | None:
    where = os.getenv("AUTOJEV_ANSWER_CACHE", "")
    if where.lower() == "off":
        return None
    return AnswerCache(Path(where) if where else _DEFAULT, fingerprint(checkpoint))
