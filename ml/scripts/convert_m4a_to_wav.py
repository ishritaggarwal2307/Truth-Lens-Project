from pydub import AudioSegment
from pathlib import Path

input_file = Path("data/audio/real/real_call.wav.m4a")
output_file = Path("data/audio/real/real_call.wav")

audio = AudioSegment.from_file(input_file, format="m4a")
audio.export(output_file, format="wav")

print("Converted to:", output_file)