# Whisper Transcription Service

A C++ service using OpenAI's Whisper model for audio transcription with a web interface for testing.

```bash
docker run -v [file directory]:/audio [container name] bash -c "cd /app && ./build/whisper_cli /audio/[file name].mp3 /audio/output.json"
```
