from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from llm_serve.errors import BadRequestError, NotFoundError
from llm_serve.schemas import BatchRecord, FileRecord


def utc_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class StorageManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files_meta_dir = root / "files" / "meta"
        self.files_content_dir = root / "files" / "content"
        self.batches_dir = root / "batches"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.files_meta_dir.mkdir(parents=True, exist_ok=True)
        self.files_content_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    def create_file(self, filename: str, purpose: str, content: bytes = b"") -> FileRecord:
        file_id = "file-" + uuid.uuid4().hex[:12]
        record = FileRecord(
            id=file_id,
            bytes=len(content),
            created_at=utc_timestamp(),
            filename=filename,
            purpose=purpose,
        )
        self._write_file_record(record)
        self._content_path(file_id).write_bytes(content)
        return record

    def append_file_content(self, file_id: str, chunk: str) -> None:
        content_path = self._content_path(file_id)
        with content_path.open("ab") as handle:
            handle.write(chunk.encode("utf-8"))
        record = self.get_file(file_id)
        record.bytes = content_path.stat().st_size
        self._write_file_record(record)

    def get_file(self, file_id: str) -> FileRecord:
        meta_path = self._meta_path(file_id)
        if not meta_path.exists():
            raise NotFoundError("File '%s' was not found" % file_id)
        return FileRecord.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))

    def get_file_content(self, file_id: str) -> bytes:
        content_path = self._content_path(file_id)
        if not content_path.exists():
            raise NotFoundError("File '%s' was not found" % file_id)
        return content_path.read_bytes()

    def create_batch(self, input_file_id: str, endpoint: str, completion_window: str) -> BatchRecord:
        output_file = self.create_file("batch-output.jsonl", "batch_output")
        error_file = self.create_file("batch-errors.jsonl", "batch_error")
        batch = BatchRecord(
            id="batch-" + uuid.uuid4().hex[:12],
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            created_at=utc_timestamp(),
            output_file_id=output_file.id,
            error_file_id=error_file.id,
        )
        self.save_batch(batch)
        return batch

    def save_batch(self, batch: BatchRecord) -> None:
        self._batch_path(batch.id).write_text(
            json.dumps(batch.model_dump(exclude_none=True), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def get_batch(self, batch_id: str) -> BatchRecord:
        batch_path = self._batch_path(batch_id)
        if not batch_path.exists():
            raise NotFoundError("Batch '%s' was not found" % batch_id)
        return BatchRecord.model_validate(json.loads(batch_path.read_text(encoding="utf-8")))

    def list_batches(self) -> List[BatchRecord]:
        batches: List[BatchRecord] = []
        for path in sorted(self.batches_dir.glob("*.json")):
            batches.append(BatchRecord.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return batches

    def _write_file_record(self, record: FileRecord) -> None:
        self._meta_path(record.id).write_text(
            json.dumps(record.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def cleanup_old_batches(self, max_age_hours: int) -> int:
        if max_age_hours <= 0:
            return 0
        now = utc_timestamp()
        cutoff = now - (max_age_hours * 3600)
        deleted = 0
        for path in list(self.batches_dir.glob("*.json")):
            try:
                batch = BatchRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if batch.completed_at is not None and batch.completed_at < cutoff:
                path.unlink(missing_ok=True)
                for file_id in [batch.output_file_id, batch.error_file_id, batch.input_file_id]:
                    if file_id:
                        self._meta_path(file_id).unlink(missing_ok=True)
                        self._content_path(file_id).unlink(missing_ok=True)
                deleted += 1
        return deleted

    def _safe_path(self, base: Path, name: str) -> Path:
        resolved = (base / name).resolve()
        if not resolved.is_relative_to(base.resolve()):
            raise BadRequestError("Invalid identifier")
        return resolved

    def _meta_path(self, file_id: str) -> Path:
        return self._safe_path(self.files_meta_dir, "%s.json" % file_id)

    def _content_path(self, file_id: str) -> Path:
        return self._safe_path(self.files_content_dir, "%s.bin" % file_id)

    def _batch_path(self, batch_id: str) -> Path:
        return self._safe_path(self.batches_dir, "%s.json" % batch_id)
