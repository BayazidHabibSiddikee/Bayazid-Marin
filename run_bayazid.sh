#!/usr/bin/env bash

# Start Bayazid server on port 5070
uvicorn main:app --host 0.0.0.0 --port 5070