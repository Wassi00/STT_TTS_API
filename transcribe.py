# transcribe.py
import whisper
import sys

audio_path = sys.argv[1]
model = whisper.load_model("large")
result = model.transcribe(audio_path, language="ar")  # or "auto" if you want automatic detection
sys.stdout.buffer.write(result["text"].encode("utf-8"))

with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
