PYTHON ?= python3

repro: data run test

data:
	$(PYTHON) scripts/download_data.py

run:
	$(PYTHON) -m driftmon.run_all

run-smoke:
	$(PYTHON) -m driftmon.run_all --smoke

test:
	$(PYTHON) -m pytest -q

tables:
	$(PYTHON) scripts/print_readme_tables.py

crosscheck:
	$(PYTHON) scripts/evidently_crosscheck.py
