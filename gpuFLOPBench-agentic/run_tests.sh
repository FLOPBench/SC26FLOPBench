#!/usr/bin/env bash

# use -vv for super verbose output, it allows us to see
# when an == assert instruction fails and prints the values/diff of values
# if it's a string
python -m pytest -vv -s ./unit-tests/check_code_search_tools.py

python -m pytest -vv -s ./unit-tests/test_cuda_kernel_uniqueness.py

python -m pytest -vv -s ./unit-tests/check_openrouter_api.py

