import re
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from transformers import pipeline


def split_into_sentences(text):
    """
    Simple regex-based sentence splitter — no NLTK required.
    Splits on '.', '?', or '!' followed by a space and a capital letter or '['.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', text)
    return [s.strip() for s in sentences if s.strip()]


def summarize_large_text(
    transcript_path,
    max_chunk_words=500,
    overlap_words=80,
    min_summary_words=100,
    max_summary_words=150,
    model_name="facebook/bart-base",
    device=-1  # set to 0 for GPU
):
    # --- Load transcript ---
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # --- Split into sentences ---
    sentences = split_into_sentences(text)
    chunks, temp, word_count = [], [], 0

    for sent in tqdm(sentences, desc="🧩 Creating chunks", unit="sentence"):
        w = len(sent.split())
        if word_count + w > max_chunk_words and temp:
            chunks.append(" ".join(temp))

            # ✅ Dynamic overlap using overlap_words
            overlap_count, overlap_sents = 0, []
            for s in reversed(temp):
                overlap_count += len(s.split())
                overlap_sents.insert(0, s)
                if overlap_count >= overlap_words:
                    break

            # Start new chunk with overlap + current sentence
            temp = overlap_sents + [sent]
            word_count = sum(len(s.split()) for s in temp)
        else:
            temp.append(sent)
            word_count += w

    if temp:
        chunks.append(" ".join(temp))

    print(f"🧩 Split into {len(chunks)} chunks with {overlap_words}-word overlap.")

    # --- Load summarization model ---
    summarizer = pipeline("summarization", model=model_name, device=device)

    # --- Summarize each chunk with progress bar ---
    summaries = []
    for chunk in tqdm(chunks, desc="🧠 Summarizing chunks", unit="chunk"):
        result = summarizer(
            chunk,
            max_length=max_summary_words,
            min_length=min_summary_words,
            do_sample=False
        )[0]['summary_text']
        summaries.append(result.strip())

    # --- Merge ---
    final_summary = "\n\n".join(summaries)

    print(f"\n✅ text summarization completed. Summary length: {len(final_summary.split())} words.")
    return final_summary


# Example usage
if __name__ == "__main__":
    summarize_large_text("../processed_audio/diarized_transcript.txt")
