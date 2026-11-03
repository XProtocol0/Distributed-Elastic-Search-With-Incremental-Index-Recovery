"""Run the coordinator: `python -m dsearch.coordinator` (configured by DSEARCH_* env vars)."""

from __future__ import annotations

import asyncio

from dsearch.config import load_config
from dsearch.coordinator.server import CoordinatorNode
from dsearch.logging_setup import configure_logging


async def main() -> None:
    config = load_config()
    configure_logging(config)
    node = await CoordinatorNode.start(config)
    try:
        await node.server.wait_for_termination()
    finally:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())
