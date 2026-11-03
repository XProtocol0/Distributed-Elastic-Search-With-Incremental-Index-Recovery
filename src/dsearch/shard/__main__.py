"""Run a shard node: `python -m dsearch.shard` (configured by DSEARCH_* env vars)."""

from __future__ import annotations

import asyncio

from dsearch.config import load_config
from dsearch.logging_setup import configure_logging
from dsearch.node_server import wait_for_shutdown_signal
from dsearch.shard.node import ShardNode


async def main() -> None:
    config = load_config()
    configure_logging(config)
    node = await ShardNode.start(config)
    try:
        await wait_for_shutdown_signal()
    finally:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())
