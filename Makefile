# HELP ########################################################################
.DEFAULT_GOAL := help

.PHONY: help
help:
	@ printf "\nusage : make <commands> \n\nthe following commands are available : \n\n"
	@ grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed -e "s/^Makefile://" | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


# APP ########################################################################

# Project settings
PROJECT := pyrl
PACKAGE := pyrl
PYTHON_VERSION=3.8

# Style makefile outputs
ECHO_COLOUR=\033[0;34m
NC=\033[0m # No Color

# Project paths
PACKAGES := $(PACKAGE)
CONFIG := $(wildcard *.py)
MODULES := $(wildcard $(PACKAGE)/*.py)

# Virtual environment paths
VIRTUAL_ENV ?= .venv

# SYSTEM DEPENDENCIES #########################################################

.PHONY: doctor
doctor:  ## Confirm system dependencies are available
	scripts/verchew --exit-code --root="${CURDIR}/scripts"

# PROJECT DEPENDENCIES ########################################################

DEPENDENCIES := $(VIRTUAL_ENV)/.poetry-$(shell scripts/checksum pyproject.toml poetry.lock)
TOOLS_FIRST_INSTALLED := $(VIRTUAL_ENV)/.tools_first_installed

.PHONY:
init: $(VIRTUAL_ENV) install-dev $(TOOLS_FIRST_INSTALLED)

.PHONY: install
install: $(DEPENDENCIES) .cache

.PHONY: install-test
install-test: install
	poetry install --with test

.PHONY: install-dev
install-dev: install
	poetry install --with dev

$(DEPENDENCIES):
	poetry install --no-root
	@ touch $@
	@ $(MAKE) gen-req

$(TOOLS_FIRST_INSTALLED): .git
	@ poetry run pre-commit install
	@ poetry run git config commit.template .gitmessage
	@ poetry self add poetry-plugin-export
	@ touch $@ # This will create a file named `.tools_first_installed` inside venv folder

.git:
	git init

.cache:
	@ mkdir -p .cache

$(VIRTUAL_ENV): ## Create python environment
	$(MAKE) doctor
	@ echo "$(ECHO_COLOUR)Configuring poetry$(NC)"
	@ poetry config virtualenvs.in-project true
	@ poetry config virtualenvs.prefer-active-python true
	@ echo "$(ECHO_COLOUR)Initializing pyenv$(NC)"
	$(eval PYENV_LATEST_VERSION=$(shell pyenv install --list | grep " $(PYTHON_VERSION)\.[0-9]*$$" | tail -1))
	@ echo "$(ECHO_COLOUR)Installing python version $(PYENV_LATEST_VERSION)...$(NC)"
	pyenv install -s $(PYENV_LATEST_VERSION)
	pyenv local $(PYENV_LATEST_VERSION)


.PHONY: gen-req
gen-req:  ## Generate requirements files from poetry
	@ echo "$(ECHO_COLOUR)Updating requirements files$(NC)"
	@ poetry export -f requirements.txt --without-hashes > requirements.txt
	@ poetry export -f requirements.txt --without-hashes --with dev > requirements-dev.txt
	@ poetry run scripts/req_fixer requirements.txt requirements-dev.txt


# CHECKS ######################################################################

.PHONY: format
format:  ## Run formatters
	poetry run isort $(PACKAGES)
	poetry run black $(PACKAGES)
	@ echo

.PHONY: check
check:  ## Run linters, and static code analysis
	poetry run safety check -r requirements.txt
	@ echo
	poetry run mypy --install-types --non-interactive $(PACKAGES)
	@ echo
	poetry run flake8 $(PACKAGES)

.PHONY: pre-commit
pre-commit:  ## Run pre-commit on all files
	poetry run pre-commit run --all-files

# CLEANUP #####################################################################

.PHONY: clean
clean: .clean-build ## Delete all generated and temporary files

.PHONY: clean-all
clean-all: clean
	rm -rf $(VIRTUAL_ENV)

.PHONY: .clean-install
.clean-install:
	find $(PACKAGES) -name '__pycache__' -delete
	rm -rf *.egg-info


# MAIN TASKS ##################################################################

.PHONY: all
all: install

.PHONY: ci
ci: format check ## Run all tasks that determine CI status

.PHONY: run
run: install ## Start the program
	poetry run python $(PACKAGE)/__main__.py
