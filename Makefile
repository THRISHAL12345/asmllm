# Makefile for asmllm
# Defines test-correctness and build targets

PYTHON ?= python

.PHONY: all test-correctness clean help

all:
	@echo "No default kernel targets built yet (Milestone M0)."

test-correctness:
	$(PYTHON) tests/correctness/test_runner.py

clean:
	rm -rf build/ __pycache__ tests/__pycache__ tests/*/__pycache__ tools/__pycache__

help:
	@echo "asmllm build and verification targets:"
	@echo "  make test-correctness   - Run numerical correctness test harness vs reference"
	@echo "  make clean              - Clean build artifacts and cache directories"
