#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
  echo ".env file not found!"
  exit 1
fi

# Export the OPENAI_API_KEY from the .env file
export $(grep -v '^#' .env | xargs)

# Verify the variable is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Failed to set OPENAI_API_KEY"
  exit 1
else
  echo "OPENAI_API_KEY set successfully"
fi