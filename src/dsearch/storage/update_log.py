"""Append-only, fsynced update log of sequenced `LogEntry` records.

On-disk record (D-008): [u32 length][u32 crc32(payload)][payload], big-endian,
where payload is a serialised `LogEntry`. Seqs are dense: each append must
carry `last_seq + 1`. A byte-offset list gives O(1) access by seq (D-009).

Blocking file I/O happens in the `*_sync` methods. Async callers use the
async wrappers, which run them in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import zlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import BinaryIO

from google.protobuf.message import DecodeError

from dsearch.proto import search_pb2

log = logging.getLogger(__name__)

_HEADER = struct.Struct(">II")  # payload length, crc32 of payload


class CorruptLogError(Exception):
    """A checksummed record holds the wrong seq: a bug, not a torn write."""


class UpdateLog:
    def __init__(self, path: Path, file: BinaryIO, offsets: list[int], end_offset: int) -> None:
        self.path = path
        self._file = file  # opened "r+b", positioned at end_offset
        self._offsets = offsets  # offsets[i] = byte offset of seq first_seq + i
        self._end_offset = end_offset
        self.first_seq = 1  # no truncation yet (D-011)

    # ---- open / close ---------------------------------------------------

    @classmethod
    def open_sync(cls, path: Path) -> "UpdateLog":
        """Open (or create) the log, dropping a torn or corrupt tail."""
        path.parent.mkdir(parents=True, exist_ok=True)
        created = not path.exists()
        if created:
            path.touch()
            _fsync_dir(path.parent)

        file = open(path, "r+b")
        offsets, valid_end = _scan(file, path)
        file_size = os.fstat(file.fileno()).st_size
        if valid_end < file_size:
            log.warning(
                "update log %s: dropping %d bytes of torn/corrupt tail after seq=%d",
                path, file_size - valid_end, len(offsets),
            )
            file.truncate(valid_end)
            os.fsync(file.fileno())
        file.seek(valid_end)
        return cls(path, file, offsets, valid_end)

    @classmethod
    async def open(cls, path: Path) -> "UpdateLog":
        return await asyncio.to_thread(cls.open_sync, path)

    def close(self) -> None:
        self._file.close()

    # ---- properties -----------------------------------------------------

    @property
    def last_seq(self) -> int:
        """Highest durable seq, or 0 if the log is empty."""
        return self.first_seq + len(self._offsets) - 1

    @property
    def size_bytes(self) -> int:
        return self._end_offset

    # ---- append ---------------------------------------------------------

    def append_sync(self, entry: search_pb2.LogEntry) -> None:
        """Write `entry` and fsync. Returns only once the entry is durable."""
        if entry.seq != self.last_seq + 1:
            raise ValueError(f"append seq={entry.seq}, expected seq={self.last_seq + 1}")
        payload = entry.SerializeToString()
        record = _HEADER.pack(len(payload), zlib.crc32(payload)) + payload
        self._file.write(record)
        self._file.flush()
        os.fsync(self._file.fileno())
        self._offsets.append(self._end_offset)
        self._end_offset += len(record)

    async def append(self, entry: search_pb2.LogEntry) -> None:
        await asyncio.to_thread(self.append_sync, entry)

    # ---- read -----------------------------------------------------------

    def read_after_sync(self, after_seq: int, max_entries: int) -> list[search_pb2.LogEntry]:
        """Up to `max_entries` entries with seq > after_seq, in seq order."""
        start_seq = max(after_seq + 1, self.first_seq)
        end_seq = min(self.last_seq, start_seq + max_entries - 1)
        if start_seq > end_seq:
            return []
        start = self._offsets[start_seq - self.first_seq]
        end = self._offset_after(end_seq)
        with open(self.path, "rb") as reader:
            reader.seek(start)
            data = reader.read(end - start)

        entries = []
        pos = 0
        while pos < len(data):
            length, _crc = _HEADER.unpack_from(data, pos)
            pos += _HEADER.size
            entry = search_pb2.LogEntry()
            entry.ParseFromString(data[pos : pos + length])
            pos += length
            entries.append(entry)
        return entries

    async def read_after(
        self, after_seq: int, batch_size: int = 1000
    ) -> AsyncIterator[search_pb2.LogEntry]:
        """Yield every entry with seq > after_seq, reading in batches off the event loop."""
        next_seq = after_seq
        while next_seq < self.last_seq:
            batch = await asyncio.to_thread(self.read_after_sync, next_seq, batch_size)
            for entry in batch:
                yield entry
            next_seq = batch[-1].seq

    def _offset_after(self, seq: int) -> int:
        index = seq - self.first_seq + 1
        return self._offsets[index] if index < len(self._offsets) else self._end_offset


def _scan(file: BinaryIO, path: Path) -> tuple[list[int], int]:
    """Return offsets of valid records and the byte offset where validity ends."""
    data = file.read()
    offsets: list[int] = []
    pos = 0
    expected_seq = 1
    while pos + _HEADER.size <= len(data):
        length, crc = _HEADER.unpack_from(data, pos)
        payload = data[pos + _HEADER.size : pos + _HEADER.size + length]
        if len(payload) < length or zlib.crc32(payload) != crc:
            break
        entry = search_pb2.LogEntry()
        try:
            entry.ParseFromString(payload)
        except DecodeError:
            break
        if entry.seq != expected_seq:
            raise CorruptLogError(
                f"{path}: found seq={entry.seq} at byte {pos}, expected seq={expected_seq}"
            )
        offsets.append(pos)
        pos += _HEADER.size + length
        expected_seq += 1
    return offsets, pos


def _fsync_dir(directory: Path) -> None:
    """Make a newly created file's directory entry durable."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
