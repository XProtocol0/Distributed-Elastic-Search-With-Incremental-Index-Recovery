PYTHON ?= .venv/bin/python

.PHONY: venv proto test test-docker corpus cluster-up cluster-down clean-proto

venv:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev,eval,data]"

proto:
	$(PYTHON) scripts/gen_proto.py

test: proto
	$(PYTHON) -m pytest

test-docker: proto
	$(PYTHON) -m pytest -m docker

corpus:
	$(PYTHON) scripts/prepare_corpus.py

cluster-up:
	docker compose up -d --build

cluster-down:
	docker compose down -v

clean-proto:
	rm -f src/dsearch/proto/*_pb2*.py src/dsearch/proto/*_pb2*.pyi
