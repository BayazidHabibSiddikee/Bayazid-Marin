#!/usr/bin/env bash

# Run Arena server (arena.py) on port 5071
uvicorn arena:app --host 0.0.0.0 --port 5071
