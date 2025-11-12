# Meeting Summarizer Demo

This small project includes a summarizer (Hugging Face transformers) and a simple
evaluator using ROUGE and BLEU. The demo runner creates a tiny
diarized transcript, runs a summarization model (`base-model` by default in the
demo), and evaluates it against a trivial reference.



Notes:
- The first run downloads model weights and may take some time.
- If you have a CUDA-enabled GPU and a suitable PyTorch install, the summarizer
  will use it automatically.

