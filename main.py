import whisperx
import gc 
import os

input_dir = "./input/Book 1 mp3/"
mp3_files = []
for file in os.listdir(input_dir):
    if file.endswith(".mp3"):
        mp3_files.append(input_dir+file)



device = "cuda" 
audio_file = mp3_files[0]
batch_size = 6 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
print("loading model...")
model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="ja")

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

print("transcribing " + audio_file)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

