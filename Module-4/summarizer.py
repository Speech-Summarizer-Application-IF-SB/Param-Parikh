# summarizer.py
"""
Summarizer module that:
 - accepts diarized JSON (list of segments)
 - builds prompts (per-template)
 - calls model (Hugging Face transformers for summarization)
 - returns structured summary (per-speaker + consolidated)
"""

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from prompts import SIMPLE_SUMMARY_PROMPT, SPEAKER_WISE_PROMPT, BULLET_PROMPT, build_diarized_text

# Config: choose model here. For local small/medium experiments use 't5-small' (faster/smaller)
DEFAULT_MODEL = "t5-small"

class HuggingFaceSummarizer:
    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        """
        Load tokenizer and model. Use device 'cuda' if GPU available.
        """
        self.model_name = model_name
        # device: -1 for CPU, 0...n for GPU ids
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        # Use transformers pipeline for summarization where possible
        # For sequence-to-sequence models we load tokenizer & model for more control
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # place model on GPU if available
        if self.device != -1:
            self.model.to("cuda")

        # pipeline wrapper (optional, cleaner API)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=self.device)

    def summarize_prompt(self, prompt_text, max_length=200, min_length=40):
        """
        Generate summary from a prompt string. Returns text.
        """
        # pipeline expects text, we can tune max_length/min_length
        out = self.summarizer(prompt_text, max_length=max_length, min_length=min_length, do_sample=False)
        # pipeline returns a list of dicts
        return out[0]["summary_text"].strip()

    def summarize_diarized(self, segments, meeting_type="meeting", template="simple", length_hint="short", max_segments=200):
        """
        High-level function:
         - Build diarized_text
         - Select prompt template
         - Call summarize
         - Optionally post-process into structured JSON
        """
        diarized_text = build_diarized_text(segments, max_segments=max_segments)

        if template == "simple":
            prompt = SIMPLE_SUMMARY_PROMPT.format(meeting_type=meeting_type, length_hint=length_hint, diarized_text=diarized_text)
            summary = self.summarize_prompt(prompt)
            return {"model": self.model_name, "type": "simple", "summary": summary, "prompt": prompt}

        elif template == "speaker_wise":
            prompt = SPEAKER_WISE_PROMPT.format(meeting_type=meeting_type, diarized_text=diarized_text)
            raw = self.summarize_prompt(prompt, max_length=400, min_length=80)
            # post-process: split lines by speaker and sections (best-effort)
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            return {"model": self.model_name, "type": "speaker_wise", "raw": raw, "lines": lines, "prompt": prompt}

        elif template == "bullets":
            prompt = BULLET_PROMPT.format(meeting_type=meeting_type, length_hint=length_hint, diarized_text=diarized_text)
            bullets = self.summarize_prompt(prompt, max_length=300, min_length=40)
            return {"model": self.model_name, "type": "bullets", "bullets": bullets, "prompt": prompt}

        else:
            raise ValueError("Unknown template: " + str(template))

if __name__ == "__main__":
    print("🔹 Summarizer module running...")

    text = """
    [Speaker 1]: Let's discuss next quarter goals.
    [Speaker 2]: We should increase sales by 20% and expand into new regions.
    [Speaker 1]: Great idea. Let's assign tasks for each department.
    """

    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarizer(text, max_length=100, min_length=25, do_sample=False)
    print("\n✅ Generated Summary:\n", summary[0]["summary_text"])
