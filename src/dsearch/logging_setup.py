"""Process-wide logging setup.

Every line carries the node id so logs from the whole cluster can be merged
and traced. Replication and recovery code logs one line per sequence number
using the `seq=<n>` key, so `grep 'shard=1 .*seq=42'` finds an entry's path.
"""

from __future__ import annotations

import logging

from dsearch.config import Config

_FORMAT = "%(asctime)s %(levelname)s node=%(node_id)s %(name)s: %(message)s"


class _NodeIdFilter(logging.Filter):
    def __init__(self, node_id: str) -> None:
        super().__init__()
        self.node_id = node_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.node_id = self.node_id
        return True


def configure_logging(config: Config) -> None:
    """Send logs to stderr at the configured level, tagged with the node id."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_NodeIdFilter(config.node_id))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(config.log_level)
