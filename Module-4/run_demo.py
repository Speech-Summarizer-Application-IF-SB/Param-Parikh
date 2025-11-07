"""run_demo.py

Small demo runner that creates a tiny diarized transcript, runs the summarizer
with a small model (t5-small) and evaluates against a trivial reference using
the existing evaluator.

Usage (from project folder):
    python -m venv .venv
    .\.venv\Scripts\Activate
    pip install -r requirements.txt
    python run_demo.py
"""

import os
from summarizer import HuggingFaceSummarizer
import evaluate_summary as evaluator


def make_demo_segments():
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 12.0, "text": "We need to finish the client report by Friday. Alice will take the lead."},
        {"speaker": "SPEAKER_01", "start": 12.0, "end": 30.0, "text": "I will prepare the first draft and share with the team. Also, schedule a review meeting."},
        {"speaker": "SPEAKER_00", "start": 30.0, "end": 60.0, "text": "Make sure to include the new metrics; Bob will collect them."},
    ]


def main():
    # use a small model for quick demo (keeps downloads small)
    summarizer = HuggingFaceSummarizer(model_name="t5-small")
    segments = make_demo_segments()
    result = summarizer.summarize_diarized(segments, meeting_type="standup", template="simple", length_hint="short")
    print("\n=== SUMMARY ===")
    print(result.get("summary") or result.get("raw") or result)

    # write demo files for evaluation
    ref_dir = "references"
    out_dir = "outputs"
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    sample_id = "demo"
    ref_path = os.path.join(ref_dir, sample_id + ".ref.txt")
    gen_path = os.path.join(out_dir, sample_id + ".gen.txt")

    # create a trivial reference (for demonstration only)
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write("Alice will lead the client report; draft due Friday. Bob will collect metrics.\n")

    with open(gen_path, "w", encoding="utf-8") as f:
        f.write(result.get("summary") if result.get("summary") else "")

    print("\n=== EVALUATION ===")
    evaluator.evaluate_folder(ref_folder=ref_dir, gen_folder=out_dir)


if __name__ == "__main__":
    main()
