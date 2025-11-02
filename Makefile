PY=python
PIP=pip

.PHONY: setup test fmt lint run docker

setup:
    $(PIP) install -r requirements.txt

fmt:
    black .
    ruff check --fix . || true

lint:
    ruff check .
    mypy --ignore-missing-imports . || true

test:
    pytest -q

run:
    $(PY) train.py --config configs/experiment.yaml --epochs 1 --fast

docker:
    docker build -t swingat:latest .
