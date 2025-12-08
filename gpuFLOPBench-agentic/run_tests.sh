#!/usr/bin/env bash

python -m pytest -v -s ./unit-tests/check_code_search_tools.py

python -m pytest -v -s ./unit-tests/check_openrouter_api.py
