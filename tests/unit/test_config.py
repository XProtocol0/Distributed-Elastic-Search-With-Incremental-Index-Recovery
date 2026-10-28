from pathlib import Path

import pytest

from dsearch.config import Config, ConfigError, Role, load_config

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "cluster.example.toml"


def test_defaults_without_file_or_env() -> None:
    assert load_config({}) == Config()


def test_env_overrides_defaults() -> None:
    config = load_config(
        {
            "DSEARCH_NODE_ID": "shard1-replica2",
            "DSEARCH_ROLE": "replica",
            "DSEARCH_SHARD_ID": "1",
            "DSEARCH_PORT": "6000",
            "DSEARCH_DATA_DIR": "/tmp/x",
            "DSEARCH_LOG_LEVEL": "debug",
            "DSEARCH_SNAPSHOT_EVERY": "500",
        }
    )
    assert config.role is Role.REPLICA
    assert config.shard_id == 1
    assert config.port == 6000
    assert config.log_level == "DEBUG"
    assert config.snapshot_every == 500
    assert config.node_data_dir == Path("/tmp/x/shard1-replica2")


def test_example_file_loads_topology() -> None:
    config = load_config({"DSEARCH_CONFIG": str(EXAMPLE)})
    assert config.num_shards == 3
    assert [s.shard_id for s in config.shards] == [0, 1, 2]
    assert config.shards[1].primary == "shard1-primary:50051"
    assert len(config.shards[2].replicas) == 2


def test_env_beats_file(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("num_shards = 4\nport = 7000\n")
    config = load_config({"DSEARCH_CONFIG": str(path), "DSEARCH_PORT": "7001"})
    assert config.num_shards == 4
    assert config.port == 7001


@pytest.mark.parametrize(
    "env",
    [
        {"DSEARCH_PORT": "abc"},
        {"DSEARCH_PORT": "70000"},
        {"DSEARCH_NUM_SHARDS": "0"},
        {"DSEARCH_SNAPSHOT_EVERY": "-1"},
        {"DSEARCH_ROLE": "leader"},
        {"DSEARCH_ROLE": "primary"},  # missing shard id
        {"DSEARCH_ROLE": "replica", "DSEARCH_SHARD_ID": "3"},  # out of range for 3 shards
    ],
)
def test_invalid_env_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        load_config(env)


def test_unknown_file_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("num_shard = 3\n")
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config({"DSEARCH_CONFIG": str(path)})


def test_topology_must_match_shard_count(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        'num_shards = 2\nreplicas_per_shard = 0\n[[shards]]\nid = 0\nprimary = "a:1"\n'
    )
    with pytest.raises(ConfigError, match="ids"):
        load_config({"DSEARCH_CONFIG": str(path)})
