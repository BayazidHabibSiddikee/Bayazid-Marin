#!/usr/bin/env bash

# Start all three FastAPI servers in parallel
./run_marin.sh &
./run_bayazid.sh &
./run_arena.sh &

# Wait for all background processes to finish (keeps the script alive)
wait