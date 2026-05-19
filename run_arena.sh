#!/usr/bin/env bash

# Start Arena server on port 5071
uvicorn arena:app --host 0.0.0.0 --port 5071