"""Deterministic serialization and atomic run publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


class OutputExistsError(FileExistsError):
    """Raised when a completed artifact would be overwritten."""


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    target.write_text(payload + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class RunDirectory:
    """Build a run in a sibling staging directory and publish on completion."""

    def __init__(self, final_path: str | Path, *, force: bool = False) -> None:
        self.final_path = Path(final_path).resolve()
        self.force = force
        self.path: Path | None = None
        self._completed = False

    def __enter__(self) -> "RunDirectory":
        if self.final_path.exists() and not self.force:
            raise OutputExistsError(f"output already exists: {self.final_path}")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(
            tempfile.mkdtemp(
                prefix=f".{self.final_path.name}.staging-",
                dir=self.final_path.parent,
            )
        )
        return self

    def _target(self, relative_path: str | Path) -> Path:
        if self.path is None:
            raise RuntimeError("RunDirectory has not been entered")
        target = (self.path / relative_path).resolve()
        if self.path not in target.parents and target != self.path:
            raise ValueError("artifact path escapes the run directory")
        return target

    def write_json(self, relative_path: str | Path, value: Any) -> None:
        write_json(self._target(relative_path), value)

    def complete(self, metadata: dict[str, Any]) -> None:
        self.write_json("COMPLETE.json", metadata)
        self._completed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del traceback
        if self.path is None:
            return
        if exc_type is not None or not self._completed:
            shutil.rmtree(self.path, ignore_errors=True)
            return
        backup: Path | None = None
        try:
            if self.final_path.exists():
                if not self.force:
                    raise OutputExistsError(f"output already exists: {self.final_path}")
                backup = self.final_path.with_name(
                    f".{self.final_path.name}.backup-{os.getpid()}"
                )
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(self.final_path, backup)
            os.replace(self.path, self.final_path)
            if backup is not None:
                shutil.rmtree(backup)
        except BaseException:
            if backup is not None and backup.exists() and not self.final_path.exists():
                os.replace(backup, self.final_path)
            raise

