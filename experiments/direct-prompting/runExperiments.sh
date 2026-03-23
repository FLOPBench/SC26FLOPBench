#!/bin/bash

python run_queries.py --trials 1 --queryBatchSize 4 --modelName anthropic/claude-opus-4.6 --maxSpend 50.0 --verbose --printPrompts
python run_queries.py --trials 1 --queryBatchSize 4 --modelName anthropic/claude-opus-4.6 --maxSpend 50.0 --verbose --printPrompts --useSASS

python run_queries.py --trials 1 --queryBatchSize 4 --modelName openai/gpt-5.4 --maxSpend 50.0 --verbose --printPrompts
python run_queries.py --trials 1 --queryBatchSize 4 --modelName openai/gpt-5.4 --maxSpend 50.0 --verbose --printPrompts --useSASS
python run_queries.py --trials 1 --queryBatchSize 4 --modelName openai/gpt-5.4 --maxSpend 50.0 --verbose --printPrompts --useIMIX
python run_queries.py --trials 1 --queryBatchSize 4 --modelName openai/gpt-5.4 --maxSpend 50.0 --verbose --printPrompts --useSASS --useIMIX
