import json
import soundfile as sf
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def sync_speakers_with_stt(audio_path, diarization_json_path):
    print("🔁 Loading diarized segments and running STT...")

    with open(diarization_json_path, "r", encoding="utf-8") as f:
        diarization_data = json.load(f)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    speaker_transcripts = []

    for seg in diarization_data:
        start_time = max(0, float(seg["start"]) - 0.2)
        end_time = min(len(audio) / sr, float(seg["end"]) + 0.2)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        slice_audio = audio[start_sample:end_sample]

        # Transcribe each segment
        inputs = processor(slice_audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].lower()

        speaker_transcripts.append({
            "speaker": seg["speaker"],
            "start": start_time,
            "end": end_time,
            "text": transcription.strip()
        })

        print(f"[{seg['speaker']}] {transcription.strip()}")

    # Save output file
    with open("output/final_synced_transcript.txt", "w", encoding="utf-8") as f:
        for s in speaker_transcripts:
            f.write(f"[{s['speaker']}]: {s['text']}\n")

    print("✅ Full transcript saved → output/final_synced_transcript.txt")

if __name__ == "__main__":
    sync_speakers_with_stt("audio/combined_meeting.wav", "output/diarized_transcript.json")
