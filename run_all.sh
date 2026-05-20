#!/usr/bin/env bash

# Start all four FastAPI servers in parallel
echo "Starting VPA servers..."

python rag_server.py &
echo "→ RAG Server (port 5080)"

uvicorn app:app --host 0.0.0.0 --port 5069 &
echo "→ Marin (port 5069)"

uvicorn main:app --host 0.0.0.0 --port 5070 &
echo "→ Bayazid (port 5070)"

uvicorn arena:app --host 0.0.0.0 --port 5071 &
echo "→ Arena (port 5071)"

echo ""
echo "All servers started!"
echo "Access at:"
echo "  - Marin:   http://localhost:5069"
echo "  - Bayazid Main Interface: http://localhost:5070"
echo "  - Arena:  http://localhost:5071"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for all background processes
wait