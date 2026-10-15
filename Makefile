PYTHON ?= .venv/bin/python

.PHONY: venv proto test clean-proto

venv:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev,eval]"

proto:
	$(PYTHON) scripts/gen_proto.py

test: proto
	$(PYTHON) -m pytest

clean-proto:
	rm -f src/dsearch/proto/*_pb2*.py src/dsearch/proto/*_pb2*.pyi
