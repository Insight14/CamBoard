# CamBoard Top-Level Makefile

PORT ?= 8000

.PHONY: help serve test clean ml-train

help:
	@echo "=================================================================="
	@echo "                      CamBoard Project Runner                     "
	@echo "=================================================================="
	@echo "  make serve          Launch interactive browser whiteboard frontend"
	@echo "  make test           Run all ONNX parity & integration unit tests"
	@echo "  make benchmark      Run in-browser WebAssembly latency benchmark"
	@echo "  make help           Show this message"
	@echo "=================================================================="

serve:
	@echo "Starting CamBoard Web UI at http://localhost:$(PORT) ..."
	python3 -m http.server $(PORT) --directory web

test:
	cd ml && make test

benchmark:
	cd ml && make benchmark

