PYTHON ?= python

verify:
	$(PYTHON) scripts/verify_release.py --check-metrics

controls:
	$(PYTHON) scripts/recompute_g2f_controls.py

analysis:
	$(PYTHON) analyses/g2f_six_models/recompute_g2f.py
