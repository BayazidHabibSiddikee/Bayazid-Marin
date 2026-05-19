#!/usr/bin/env bash

# Start Marin server on port 5069
uvicorn app:app --host 0.0.0.0 --port 5069