#!/bin/bash

python run_voting_queries.py --trials 3 --singleDryRun --modelNames "openai/gpt-oss-120b,openai/gpt-5.4-nano,google/gemini-3-flash-preview" --queryBatchSize 9

python run_voting_queries.py --trials 3 --singleDryRun --modelNames "google/gemini-3.1-flash-lite-preview,openai/gpt-5.4-nano,google/gemini-3-flash-preview" --queryBatchSize 9
python run_voting_queries.py --trials 3 --singleDryRun --modelNames "google/gemini-3.1-flash-lite-preview,openai/gpt-5.4-nano,google/gemini-3-flash-preview,nvidia/nemotron-3-nano-30b-a3b" --queryBatchSize 9