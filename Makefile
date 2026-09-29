# Train/inference mismatch. `make help` for the stage list.
SHELL := /bin/bash
PY ?= $(shell . ./env.sh 2>/dev/null; echo $${TIM_PY:-python})

.PHONY: help test test-fast sample score report measure grpo clean

help:
	@echo "CPU:  make test | make test-fast | make report"
	@echo "GPU (via slurm/submit.sh):"
	@echo "  make measure MODEL=Qwen/Qwen3-1.7B   stages 1-3 in one job"
	@echo "  make grpo ARM=none                   stage 4 (see scripts/04_grpo.sh)"

test:
	@rc=0; for t in tests/test_repo_integrity.py tests/test_metrics.py tests/test_store.py tests/test_score_hf_tiny.py; do \
	  echo; echo "===== $$t ====="; $(PY) $$t || rc=1; done; \
	echo; if [ $$rc -ne 0 ]; then echo "SOME TESTS FAILED"; else echo "ALL TESTS PASSED"; fi; exit $$rc

test-fast:
	@$(PY) tests/test_repo_integrity.py && $(PY) tests/test_metrics.py

MODEL ?= Qwen/Qwen3-1.7B
measure: ; slurm/submit.sh --job measure_$(shell echo $(MODEL) | sed 's#.*/##; s/\./p/g') --gpus 1 --requeue -- scripts/run_measure.sh $(MODEL)
report:  ; bash scripts/03_report.sh $(MODEL)
clean:   ; find . -name __pycache__ -type d -prune -exec rm -rf {} +
