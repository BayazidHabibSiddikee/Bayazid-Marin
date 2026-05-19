#!/usr/bin/env bash

# Run Marin server (app.py) on port 5069
uvicorn app:app --host 0.0.0.0 --port 5069
