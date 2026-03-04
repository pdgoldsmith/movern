#!/usr/bin/env bash

set -e
set -x

PYTHONPATH=movern pytest \
  --junitxml=junit.xml \
  --cov-config=.coveragerc \
  --cov-report=term-missing \
  --cov=movern tests/ "${@}"
