from pyannote.audio import Pipeline
import json
import os

# put your real huggingface token below
HF_TOKEN = "HuggingFace Token"

def run_diarization(input_path):
    print("🎧 Running speaker diarization ...")

    # load the pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=HF_TOKEN)

    # force exactly 2 speakers to match your dataset
    diarization = pipeline({"uri": "meeting", "audio": input_path},
                           min_speakers=2, max_speakers=2)

    # convert to list of segments
    results = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "speaker": speaker
        })

    os.makedirs("output", exist_ok=True)
    with open("output/diarized_transcript.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("✅ Diarization done → output/diarized_transcript.json")

if __name__ == "__main__":
    run_diarization("audio/combined_meeting.wav")
