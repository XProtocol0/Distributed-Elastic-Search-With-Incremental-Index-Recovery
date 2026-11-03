# Distributed-Elastic-Search-With-Incremental-Index-Recovery

## Quick start (Docker Compose)

```sh
make venv                      # Python 3.12 venv with dev, eval and data extras
make corpus                    # data/corpus.jsonl: 100k Simple English Wikipedia abstracts
make cluster-up                # 1 coordinator + 3 shards x (1 primary + 2 replicas)

docker compose run --rm tools python scripts/load_data.py --limit 20000
docker compose run --rm tools python scripts/run_workload.py --count 10000 --rate 300 &
python scripts/inject_fault.py disconnect shard0-replica1   # partition a replica
python scripts/inject_fault.py kill shard1-replica2         # crash another
sleep 15                                                    # churn continues without them
python scripts/inject_fault.py connect shard0-replica1
python scripts/inject_fault.py start shard1-replica2
wait

docker compose run --rm tools python scripts/cluster_status.py --wait-replicas 60
docker compose run --rm tools python scripts/check_convergence.py --ops data/run/ops.jsonl
python scripts/run_queries.py "solar system" --k 5           # coordinator on localhost:50050
make cluster-down              # stop and delete all node data
```

`python scripts/fault_scenario.py --fresh` runs the whole partition-and-crash scenario and ends with the
convergence check. `make test` runs the test suite without Docker; `make test-docker` runs the Docker scenario.
