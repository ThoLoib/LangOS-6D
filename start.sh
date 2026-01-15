#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for server to be ready
echo "Waiting for Ollama server..."
until curl -s http://localhost:11434 > /dev/null; do
  sleep 1
done

# Pull the model
ollama pull mistral-small3.1

# Keep the container alive
tail -f /dev/null
