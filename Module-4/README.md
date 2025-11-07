# Meeting Summarizer Demo

This small project includes a summarizer (Hugging Face transformers) and a simple
evaluator using ROUGE and BLEU. The demo runner `run_demo.py` creates a tiny
diarized transcript, runs a summarization model (`t5-small` by default in the
demo), and evaluates it against a trivial reference.

Quick start (Windows PowerShell):

```powershell
cd "C:\Users\lenovo\Desktop\Infosys Internship Project\Milestone-2\Module-4"
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python run_demo.py
```

Notes:
- The first run downloads model weights and may take some time.
- If you have a CUDA-enabled GPU and a suitable PyTorch install, the summarizer
  will use it automatically.
