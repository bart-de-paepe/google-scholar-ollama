#!/bin/bash

# Start Ollama in the background.
ollama serve &
# Record Process ID.
pid=$!

# Pause to allow Ollama to start.
sleep 5

echo "🔴 Retrieve gemma3:1b model..."
ollama pull gemma3:27b
echo "🟢 Done!"

# Wait for the Ollama process to finish.
wait $pid