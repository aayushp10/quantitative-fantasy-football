PY := .venv/bin/python
SHELL := /bin/bash

.PHONY: web-data web-api web-ui web web-test help

help:
	@echo "make web-data   build webapp/data artifacts (needs the parquet cache;"
	@echo "                first run downloads ~2-3 GB and retrains — slow is normal)"
	@echo "make web-api    FastAPI on :8000"
	@echo "make web-ui     Vite dev server on :5173 (proxies /api -> :8000)"
	@echo "make web        both, with one Ctrl-C to stop"
	@echo "make web-test   engine + API tests (pytest) and tsc --noEmit"

web-data:
	$(PY) scripts/build_web_data.py --season 2026 --out webapp/data/

web-api:
	$(PY) -m uvicorn webapp.api.main:app --port 8000

web-ui:
	cd webapp/ui && npm run dev

web:
	@echo "=============================================="
	@echo "  FF/TERMINAL — api :8000, ui :5173"
	@echo "  open http://localhost:5173"
	@echo "=============================================="
	@trap 'kill 0' INT TERM; \
	$(PY) -m uvicorn webapp.api.main:app --port 8000 & \
	( cd webapp/ui && npm run dev ) & \
	wait

web-test:
	$(PY) -m pytest tests/test_draft_engine.py tests/test_web_api.py -q
	cd webapp/ui && npx tsc --noEmit
