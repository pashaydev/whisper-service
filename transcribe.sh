#!/bin/bash
# transcribe.sh

if [ -z "$1" ]; then
  echo "Usage: ./transcribe.sh <audio_file>"
  exit 1
fi

AUDIO_FILE="$1"

if [ ! -f "$AUDIO_FILE" ]; then
  echo "Error: File not found: $AUDIO_FILE"
  exit 1
fi

curl -X POST -F "audio=@$AUDIO_FILE" http://localhost:8080/api/transcribe
