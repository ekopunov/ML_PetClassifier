# Check if Python 3.9 is available
PYTHON_VERSION := $(shell python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

# Name of the virtual environment
VENV_NAME := myenv

# Name of the requirements file
REQUIREMENTS := requirements.txt

.PHONY: install

install: check_python install_requirements

check_python:
	@if [ "$(PYTHON_VERSION)" != "3.9" ]; then \
        echo "Error: Python 3.9 is required."; \
        exit 1; \
    fi

install_requirements: 
	pip install -r $(REQUIREMENTS) && \
	pip install src/adoption_predictor && \
	pip install src/gcloud_downloader

install_requirements: 
	pip install -r $(REQUIREMENTS) && \
	pip install -e src/adoption_predictor && \
	pip install -e src/gcloud_downloader
