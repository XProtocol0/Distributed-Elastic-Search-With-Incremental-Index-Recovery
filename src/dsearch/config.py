"""Node and cluster configuration.

Settings come from an optional TOML file (path in DSEARCH_CONFIG) holding the
shared cluster topology, overridden per node by DSEARCH_* environment
variables. Precedence: env > file > defaults (see D-002).
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Role(str, Enum):
    COORDINATOR = "coordinator"
    PRIMARY = "primary"
    REPLICA = "replica"


@dataclass(frozen=True)
class ShardTopology:
    """Addresses (host:port) of one shard's primary and replicas."""

    shard_id: int
    primary: str
    replicas: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    node_id: str = "node-0"
    role: Role = Role.COORDINATOR
    shard_id: int | None = None
    num_shards: int = 3
    replicas_per_shard: int = 2
    host: str = "127.0.0.1"
    port: int = 50051
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    snapshot_every: int = 10_000  # applied entries between snapshots; 0 = off (D-011)
    shards: tuple[ShardTopology, ...] = field(default_factory=tuple)

    @property
    def listen_address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def node_data_dir(self) -> Path:
        """Per-node directory for the update log and SQLite state."""
        return self.data_dir / self.node_id


class ConfigError(ValueError):
    pass


# Env var name -> (config field, parser). Topology is file-only.
_ENV_FIELDS: dict[str, tuple[str, Any]] = {
    "DSEARCH_NODE_ID": ("node_id", str),
    "DSEARCH_ROLE": ("role", Role),
    "DSEARCH_SHARD_ID": ("shard_id", int),
    "DSEARCH_NUM_SHARDS": ("num_shards", int),
    "DSEARCH_REPLICAS_PER_SHARD": ("replicas_per_shard", int),
    "DSEARCH_HOST": ("host", str),
    "DSEARCH_PORT": ("port", int),
    "DSEARCH_DATA_DIR": ("data_dir", Path),
    "DSEARCH_LOG_LEVEL": ("log_level", str.upper),
    "DSEARCH_SNAPSHOT_EVERY": ("snapshot_every", int),
}

_FILE_FIELDS: dict[str, Any] = {name: parse for name, parse in _ENV_FIELDS.values()}


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build the config from the optional TOML file and DSEARCH_* env vars."""
    env = os.environ if env is None else env
    values: dict[str, Any] = {}

    config_path = env.get("DSEARCH_CONFIG")
    if config_path:
        values.update(_read_file(Path(config_path)))

    for var, (name, parse) in _ENV_FIELDS.items():
        if var in env and env[var] != "":
            values[name] = _parse(var, parse, env[var])

    config = Config(**values)
    _validate(config)
    return config


def _read_file(path: Path) -> dict[str, Any]:
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc

    values: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "shards":
            values["shards"] = tuple(_parse_shard(s) for s in value)
        elif key in _FILE_FIELDS:
            values[key] = _parse(key, _FILE_FIELDS[key], value)
        else:
            raise ConfigError(f"unknown config key {key!r} in {path}")
    return values


def _parse_shard(raw: Mapping[str, Any]) -> ShardTopology:
    try:
        return ShardTopology(
            shard_id=int(raw["id"]),
            primary=str(raw["primary"]),
            replicas=tuple(str(r) for r in raw.get("replicas", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid [[shards]] entry {raw!r}: {exc}") from exc


def _parse(name: str, parse: Any, value: Any) -> Any:
    try:
        return parse(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid value {value!r} for {name}") from exc


def _validate(config: Config) -> None:
    if config.num_shards < 1:
        raise ConfigError("num_shards must be at least 1")
    if config.replicas_per_shard < 0:
        raise ConfigError("replicas_per_shard must not be negative")
    if config.snapshot_every < 0:
        raise ConfigError("snapshot_every must not be negative")
    if not 0 < config.port < 65536:
        raise ConfigError(f"port {config.port} out of range")
    if config.role in (Role.PRIMARY, Role.REPLICA):
        if config.shard_id is None:
            raise ConfigError(f"role {config.role.value} needs DSEARCH_SHARD_ID")
        if not 0 <= config.shard_id < config.num_shards:
            raise ConfigError(f"shard_id {config.shard_id} not in [0, {config.num_shards})")
    if config.shards:
        ids = sorted(s.shard_id for s in config.shards)
        if ids != list(range(config.num_shards)):
            raise ConfigError(f"[[shards]] ids {ids} must be exactly 0..{config.num_shards - 1}")
        for shard in config.shards:
            if len(shard.replicas) != config.replicas_per_shard:
                raise ConfigError(
                    f"shard {shard.shard_id} lists {len(shard.replicas)} replicas, "
                    f"expected {config.replicas_per_shard}"
                )
