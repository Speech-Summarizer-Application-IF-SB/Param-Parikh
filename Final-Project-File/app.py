
import os
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
from scipy import signal
# from openai import OpenAI  # ← NEW: Import this way
import tempfile
import json
import glob
import re
from datetime import datetime
from collections import Counter
import whisper
import io
import streamlit as st
import sounddevice as sd
# 👇 this must come immediately after the import
st.set_page_config(page_title="Meeting Summarizer (Milestone 4)", layout="wide")
import uuid
import smtplib
from email.message import EmailMessage
import threading
import time
import pandas as pd
# import soundfile as sf
import numpy as np
from pathlib import Path


# === Canonical project paths ===
BASE_DIR    = Path(__file__).parent.resolve()
UPLOAD_DIR  = BASE_DIR / "uploaded_audio"
SESSIONS_DIR= BASE_DIR / "sessions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file) -> str:
    """
    Save upload exactly once, return absolute path, and remember it in session state.
    """
    ext = Path(uploaded_file.name).suffix.lower() or ".wav"
    fname = f"{uuid.uuid4().hex}{ext}"
    dest  = UPLOAD_DIR / fname
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["uploaded_audio_path"] = str(dest)   # single source of truth
    return st.session_state["uploaded_audio_path"]

def require_audio_path() -> str:
    """
    Read the same path we saved. Stop the app if it’s missing.
    """
    p = st.session_state.get("uploaded_audio_path")
    if not p or not os.path.exists(p):
        st.error(f"No audio file available. Expected: {p or '(unset)'}")
        st.stop()
    return p


# Try to import reportlab (preferred) or fpdf as a fallback; set a flag we can check later.
REPORTLAB_AVAILABLE = False
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except Exception:
    try:
        from fpdf import FPDF
        # Use a sentinel value to indicate FPDF fallback
        REPORTLAB_AVAILABLE = "FPDF"
    except Exception:
        REPORTLAB_AVAILABLE = False


# Load environment variables
load_dotenv()

# import smtplib
# from email.message import EmailMessage
# from reportlab.lib.pagesizes import letter



# Instead, load from .env safely:
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create persistent directories
os.makedirs("uploaded_audio", exist_ok=True)
os.makedirs("sessions", exist_ok=True)



# Initialize OpenAI client (NEW FORMAT)
# from openai import OpenAI

# Initialize session state
for key, default in {
    'recording': False,
    'transcript': "",
    'summary': "",
    'live_transcript': "",
    'audio_recorder': None,
    'uploaded_audio_path': None,
    'uploaded_audio': None,
    'saved_sessions': [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def save_uploaded_audio_to_disk(uploaded_file):
    """Save uploaded file to persistent location to prevent WinError 2."""
    try:
        # Use permanent folder, not temp
        filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        permanent_path = os.path.join("uploaded_audio", filename)
        
        # Write file immediately
        with open(permanent_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return permanent_path
    except Exception as e:
        st.error(f"❌ File save failed: {e}")
        return None



# Define transcribe function
import whisper

# Load model once
@st.cache_resource
def load_whisper_model():
    import torch
    return whisper.load_model("base", device="cpu", download_root=None, in_memory=False)

model = load_whisper_model()

def transcribe_with_local_whisper(uploaded_audio_path: str) -> str:
    """Transcribe using local Whisper model (no API needed)."""
    try:
        # FIX: Verify file exists
        if not os.path.exists(uploaded_audio_path):
            return f"ERROR: File not found: {uploaded_audio_path}"
        
        if not os.path.isfile(uploaded_audio_path):
            return f"ERROR: Not a valid file: {uploaded_audio_path}"
        
        # Load model
        if model is None:
            return "ERROR: Whisper model not loaded"
        
        # Transcribe
        with st.spinner("Transcribing..."):
            result = model.transcribe(uploaded_audio_path, language="en")
        
        transcript_text = result.get("text", "").strip()
        if not transcript_text:
            return "ERROR: No speech detected in audio"
        
        return transcript_text
    except Exception as e:
        return f"ERROR: {str(e)}"




# # Initialize session state
# for key, default in {
#     'recording': False,
#     'transcript': "",
#     'summary': "",
#     'live_transcript': "",
#     'audio_recorder': None,
#     'uploaded_audio_path': None,
#     'uploaded_audio': None,   # <-- add this line
#     'saved_sessions': [],
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = default


# Load Whisper model once
# @st.cache_resource
# def load_whisper_model():
#     return whisper.load_model("base")

# model = load_whisper_model()

# ... rest of your code ...

def load_audio(path_or_array):
    if isinstance(path_or_array, str):
        audio, sr = sf.read(path_or_array)
    else:
        audio, sr = path_or_array, 16000
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    return audio.astype(np.float32)


# ============ TRANSCRIPTION FUNCTION ============
def transcribe_with_openai(uploaded_audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper API (v1.0+ format)."""
    try:
        with open(uploaded_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        st.error(f"OpenAI Whisper API failed: {e}")
        return ""


def transcribe_audio(audio_input):
    audio = load_audio(audio_input)
    result = model.transcribe(audio)
    return result["text"]

# def summarize_text(text):
#     if not openai.api_key:
#         return "No OpenAI key provided."
#     resp = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role":"user","content":f"Summarize this meeting transcript:\n\n{text}"}],
#         max_tokens=300
#     )
#     return resp.choices[0].message.content.strip()


from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline("text2text-generation", model="t5-small", tokenizer="t5-small")

def summarize_locally(text: str) -> str:
    if not text.strip():
        return ""
    prompt = "summarize: " + text.strip()
    out = load_summarizer()(prompt, max_length=180, min_length=48, do_sample=False)
    return out[0]["generated_text"].strip()

from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline("text2text-generation", model="t5-small", tokenizer="t5-small")

def summarize_locally(text: str) -> str:
    if not text.strip():
        return ""
    prompt = "summarize: " + text.strip()
    out = load_summarizer()(prompt, max_length=180, min_length=48, do_sample=False)
    return out[0]["generated_text"].strip()


# import streamlit as st

# ─── Session State Initialization ────────────────────────────────────────────
# Initialize all needed session-state keys
# for key, default in {
#     'recording': False,
#     'transcript': "",
#     'summary': "",
#     'live_transcript': "",
#     'audio_recorder': None,
#     'uploaded_audio_path': None,
#     'saved_sessions': [],
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = default
# ────────────────────────────────────────────────────────────────────────────────


# Add this block at the beginning of your app.py, after imports
if 'recording' not in st.session_state:
    st.session_state['recording'] = False

if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""

if 'audio_buffer' not in st.session_state:
    st.session_state['audio_buffer'] = []

if 'live_transcript' not in st.session_state:
    st.session_state['live_transcript'] = ""

if 'audio_queue' not in st.session_state:
    st.session_state['audio_queue'] = []


load_dotenv()  # Load .env file containing GROQ_API_KEY and HF_TOKEN

USE_FASTER_WHISPER = False
try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
except Exception:
    try:
        import whisper as openai_whisper
    except Exception:
        openai_whisper = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:
    PyannotePipeline = None

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
CHUNK_FRAMES = SAMPLE_RATE * CHUNK_SECONDS
HF_TOKEN = os.getenv("HF_TOKEN", "")
SESSIONS_DIR = "sessions"

for key, default in {
    'recording': False,
    'audio_chunks': [],
    'live_segments': [],
    'diarized_transcript': None,
    'summary': None,
    'status': "Idle",
    'summ_pipes': {},
    'models_used': {},
    'metrics': {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource(show_spinner=False)
def load_models():
    models = {"stt_model": None, "stt_type": None}
    if USE_FASTER_WHISPER:
        try:
            import torch
            device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
            models["stt_model"] = WhisperModel("small", device=device)
            models["stt_type"] = "faster-whisper"
            models["stt_name"] = "faster-whisper-small"
        except Exception:
            pass
    if models["stt_model"] is None and 'openai_whisper' in globals() and openai_whisper is not None:
        try:
            models["stt_model"] = openai_whisper.load_model("small")
            models["stt_type"] = "openai-whisper"
            models["stt_name"] = "openai-whisper-small"
        except Exception:
            pass
    return models

_models = load_models()
stt_model, stt_type = _models.get("stt_model"), _models.get("stt_type")
if _models.get("stt_name"):
    st.session_state['models_used']['stt'] = _models.get("stt_name")

# Truncate text at nearest sentence break under max_chars
def get_truncated_text(text, max_chars=8000):
    if len(text) <= max_chars:
        return text
    sentences = re.split(r'(?<=[.!?]) +', text)
    out = ""
    for sentence in sentences:
        if len(out) + len(sentence) > max_chars:
            break
        out += sentence + " "
    return out.strip()

def summarize_with_groq(text):
    safe_char_limit = 8000
    prompt_text = get_truncated_text(text, max_chars=safe_char_limit)
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Groq API key not found. Please set GROQ_API_KEY in .env"
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert meeting summarization assistant."},
                {"role": "user", "content": f"Summarize this meeting transcript concisely:\n\n{prompt_text}"}
            ],
            max_tokens=600,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq summarization failed: {e}"

def transcribe_file(path, model, model_type):
    out = []
    if not model:
        return out
    try:
        if model_type == "faster-whisper":
            segments, _ = model.transcribe(path, language="en", beam_size=5)
            for seg in segments:
                out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
        elif model_type == "openai-whisper":
            res = model.transcribe(path, language="en", verbose=False)
            for seg in res.get("segments", []):
                out.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()})
    except Exception:
        pass
    return out

def get_summarizer_pipeline(model_name):
    pipes = st.session_state.get('summ_pipes', {})
    if model_name in pipes and pipes[model_name] is not None:
        return pipes[model_name]
    try:
        pipe = pipeline("summarization", model=model_name)
        pipes[model_name] = pipe
        st.session_state['summ_pipes'] = pipes
        st.session_state['models_used'].setdefault('summarizers', []).append(model_name)
        return pipe
    except Exception as e:
        pipes[model_name] = None
        st.session_state['summ_pipes'] = pipes
        st.session_state['status'] = f"Summarizer load failed for {model_name}: {e}"
        return None

def tokenize_simple(text):
    return re.findall(r"\w+", (text or "").lower())

def rouge1_f1(reference, hypothesis):
    r_tokens = tokenize_simple(reference)
    h_tokens = tokenize_simple(hypothesis)
    if not r_tokens or not h_tokens:
        return 0.0
    r_counts = Counter(r_tokens)
    h_counts = Counter(h_tokens)
    overlap = sum(min(r_counts[t], h_counts.get(t, 0)) for t in r_tokens)
    precision = overlap / max(1, sum(h_counts.values()))
    recall = overlap / max(1, sum(r_counts.values()))
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return min(f1 * 100.0, 100.0)

def summarize_text(text, model_name):
    if not text:
        return ""
    if model_name == "groqai/groq-summarizer":
        return summarize_with_groq(text)
    pipe = get_summarizer_pipeline(model_name)
    if pipe is None:
        return text[:500] + ("..." if len(text) > 500 else "")
    max_chars = 1200
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    parts = []
    for chunk in chunks:
        try:
            result = pipe(chunk, max_length=130, min_length=30, do_sample=False)
            parts.append(result[0]['summary_text'])
        except Exception:
            parts.append(chunk[:200] + "...")
    return " ".join(parts)
def create_pdf_bytes(text, title="Meeting Summary"):
    # reportlab first
    if REPORTLAB_AVAILABLE is True:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        y = 750
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, title); y -= 24
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Exported: {datetime.utcnow().isoformat()}Z"); y -= 20
        for line in text.splitlines():
            while len(line) > 110:
                c.drawString(40, y, line[:110]); line = line[110:]; y -= 12
                if y < 40: c.showPage(); y = 750
            c.drawString(40, y, line); y -= 12
            if y < 40: c.showPage(); y = 750
        c.save(); buf.seek(0)
        return buf.getvalue()

    # FPDF fallback
    if REPORTLAB_AVAILABLE == "FPDF":
        from fpdf import FPDF
        pdf = FPDF(); pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Exported: {datetime.utcnow().isoformat()}Z", ln=True)
        pdf.ln(4)
        for line in text.splitlines():
            pdf.multi_cell(0, 6, line)
        return pdf.output(dest='S').encode('latin-1')

    # No PDF lib: return text bytes (safe fallback)
    fallback = f"{title}\n\nExported: {datetime.utcnow().isoformat()}Z\n\n{text}"
    return fallback.encode("utf-8")



def align_diarization(stt_segments, diar_segments):
    aligned = []
    for d in diar_segments:
        s, e, sp = d['start'], d['end'], d['speaker']
        texts, seg_start, seg_end = [], None, None
        for t in stt_segments:
            if (t['end'] > s) and (t['start'] < e):
                texts.append(t['text'])
                seg_start = min(seg_start or t['start'], t['start'])
                seg_end = max(seg_end or t['end'], t['end'])
        if texts:
            aligned.append({"speaker": sp, "start": seg_start or s, "end": seg_end or e, "text": " ".join(texts)})
    return aligned

def compute_duration_from_segments(segments):
    if not segments:
        return 0.0
    start = min(s['start'] for s in segments)
    end = max(s['end'] for s in segments)
    return max(0.0, end - start)

def capture_worker(stop_event, chunk_queue):
    buffer = np.zeros((0, 1), dtype='float32')
    idx = 0
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=1024, latency='low') as stream:
            while not stop_event.is_set():
                frames, _ = stream.read(1024)
                buffer = np.vstack([buffer, frames])
                while buffer.shape[0] >= CHUNK_FRAMES:
                    chunk = buffer[:CHUNK_FRAMES].copy()
                    buffer = buffer[CHUNK_FRAMES:]
                    chunk_queue.put((chunk, idx))
                    st.session_state['audio_chunks'].append(chunk)
                    idx += 1
                time.sleep(0.01)
            if buffer.shape[0] > 0:
                chunk_queue.put((buffer.copy(), idx))
                st.session_state['audio_chunks'].append(buffer.copy())
    except Exception as e:
        st.session_state['status'] = f"Recording failed: {e}"
    finally:
        chunk_queue.put(None)

def stt_worker(chunk_queue, done_event):
    while True:
        item = chunk_queue.get()
        if item is None:
            chunk_queue.task_done()
            break
        chunk, idx = item
        tmp = os.path.join(tempfile.gettempdir(), f"chunk_{uuid.uuid4().hex}.wav")
        try:
            sf.write(tmp, chunk, SAMPLE_RATE, format='WAV')
            segs = transcribe_file(tmp, stt_model, stt_type)
            for s in segs:
                st.session_state['live_segments'].append({
                    "start": s['start'] + idx * CHUNK_SECONDS,
                    "end": s['end'] + idx * CHUNK_SECONDS,
                    "text": s['text']
                })
        except Exception:
            pass
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
        chunk_queue.task_done()
    done_event.set()

def backend_pipeline(wav_path, use_diar):
    try:
        if 'metrics' not in st.session_state or not isinstance(st.session_state['metrics'], dict):
            st.session_state['metrics'] = {}
        st.session_state['status'] = "Transcribing"
        segs = transcribe_file(wav_path, stt_model, stt_type)
        if not segs:
            st.session_state['status'] = "No transcription output"
            return
        st.session_state['live_segments'] = [{"start": s['start'], "end": s['end'], "text": s['text']} for s in segs]
        diarized_lines = []
        if use_diar and PyannotePipeline:
            st.session_state['status'] = "Diarizing"
            try:
                diarizer = PyannotePipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN or None)
                diar = diarizer(wav_path)
                diar_segments = [{"start": turn.start, "end": turn.end, "speaker": label} for turn, _, label in diar.itertracks(yield_label=True)]
                diarized_lines = align_diarization(st.session_state['live_segments'], diar_segments)
                st.session_state['models_used']['diarization'] = "pyannote/speaker-diarization"
            except Exception as e:
                st.session_state['status'] = f"Diarization failed: {e}"
        if not diarized_lines:
            merged = " ".join([s['text'] for s in st.session_state['live_segments']])
            diarized_lines = [{"speaker": "SPEAKER_1", "start": 0.0, "end": st.session_state['live_segments'][-1]['end'] if st.session_state['live_segments'] else 0.0, "text": merged}]
        st.session_state['diarized_transcript'] = diarized_lines
        st.session_state['status'] = "Summarizing"
        full_text = "\n".join([f"[{l['speaker']}] {l['start']:.1f}-{l['end']:.1f}: {l['text']}" for l in diarized_lines])
        model_name = st.session_state.get('summ_choice', 'facebook/bart-large-cnn')
        summary = summarize_text(full_text, model_name)
        if not summary:
            st.session_state['status'] = "Summary generation failed"
            return
        st.session_state['summary'] = summary
        rouge_score = rouge1_f1(full_text, summary)
        st.session_state['summary_accuracy'] = rouge_score
        duration = compute_duration_from_segments(st.session_state['diarized_transcript'])
        st.session_state['metrics']['duration_seconds'] = duration
        st.session_state['metrics']['rouge1_f1'] = rouge_score
        st.session_state['status'] = "Done"
        st.session_state['backend_done'] = True
    except Exception as e:
        st.session_state['status'] = f"Error: {e}"

if st.session_state.get('backend_done'):
    st.session_state['backend_done'] = False
    st.rerun()

def safe_filename(s):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)

def save_session_json(base_dir, session_data):
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    title = safe_filename(session_data.get("title", "session"))
    fname = os.path.join(base_dir, f"{title}_{ts}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    return fname

def save_session_parquet(base_dir, session_data):
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    title = safe_filename(session_data.get("title", "session"))
    df = pd.DataFrame([session_data])
    fname = os.path.join(base_dir, f"{title}_{ts}.parquet")
    df.to_parquet(fname, index=False, compression='snappy')
    return fname

def list_session_files(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(base_dir, "*")), reverse=True)
    return files

def load_json_session(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# st.set_page_config(page_title="Meeting Summarizer (Milestone 4)", layout="wide")
st.markdown("""
<style>
    :root { --bg: #0f1720; --card: #0f1726; --muted: #8b94a6; --accent: #ff8c3a; --panel-border: rgba(255,255,255,0.03);} 
    html, body, [class*="css"]  { background: var(--bg) !important; color: #e6eef8; }
    .title { font-size:34px; font-weight:700; text-align:center; margin-bottom:8px; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:18px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); border:1px solid var(--panel-border); }
    .small-muted { color: var(--muted); font-size:13px; }
    .btn-orange { background: linear-gradient(180deg,#ff9c4a,#ff7b2a); color:#08121a; padding:10px 18px; border-radius:8px; font-weight:700; border:none; }
    .btn-ghost { background: transparent; color: #cbd5e1; padding:10px 18px; border-radius:8px; border:1px solid rgba(255,255,255,0.04);} 
    .uploader { border-radius:8px; padding:8px; background: rgba(255,255,255,0.01); }
    .summary-box { background: rgba(255,255,255,0.01); border-radius:8px; padding:12px; min-height:260px; }
    textarea[readonly] { background: transparent; color: #e6eef8; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Meeting Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="small-muted" style="text-align:center;margin-bottom:18px;">Upload an audio file or record live. Get diarized transcript, concise summary, export & storage.</div>', unsafe_allow_html=True)

cols = st.columns([1, 2, 1])
with cols[1]:
    st.markdown(f"**Status:** <span style='color:#cbd5e1;'>{st.session_state['status']}</span>", unsafe_allow_html=True)

left, right = st.columns([6, 4], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h4>Audio</h4>', unsafe_allow_html=True)
    
    meeting_title = st.text_input(
        "Meeting Title",
        value="Meeting",
        placeholder="Enter meeting title...",
        help="This will be used for the filename when exporting"
    )

    uploaded_file = st.file_uploader("Upload an audio file", type=['wav','mp3','m4a'], key='uploader')

    if uploaded_file is not None:
        # Save file immediately when uploaded
        if 'last_uploaded_name' not in st.session_state or st.session_state['last_uploaded_name'] != uploaded_file.name:
            try:
                saved_path = save_uploaded_file(uploaded_file)
                st.session_state['uploaded_audio_path'] = saved_path
                st.session_state['last_uploaded_name'] = uploaded_file.name
                st.audio(saved_path)
                st.success(f"✅ Saved: {Path(saved_path).name}")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")
        else:
            # File already saved, just display it
            saved_path = st.session_state.get('uploaded_audio_path')
            if saved_path and os.path.exists(saved_path):
                st.audio(saved_path)


    import whisper
    model = whisper.load_model("base")

# Transcribe button
    if st.button("🔍 Transcribe"):
        uploaded_audio_path = require_audio_path()
        
        # Debug output
        st.write(f"🔍 DEBUG: audio_path = `{uploaded_audio_path}`")
        st.write(f"🔍 DEBUG: exists? {os.path.exists(uploaded_audio_path)}")
        
        if not uploaded_audio_path.lower().endswith(".wav"):
            st.error("Please upload a WAV file.")
        else:
            with st.spinner("Transcribing..."):
                out = transcribe_with_local_whisper(uploaded_audio_path)
            if out.startswith("ERROR:"):
                st.error(out)
            else:
                st.session_state["transcript"] = out
                st.success("✅ Transcription complete!")
                st.text_area("Transcript:", out, height=200)
                
                # Trigger backend pipeline for summarization
                st.session_state['live_segments'] = [{"start": 0.0, "end": 0.0, "text": out}]
                use_diar = st.session_state.get('use_diar', False)
                threading.Thread(target=backend_pipeline, args=(uploaded_audio_path, use_diar), daemon=True).start()


    st.markdown('---')

    st.markdown('<div class="small-muted">Live Recording</div>', unsafe_allow_html=True)
    from fix_audio import AudioRecorder

# Initialize audio recorder
if 'audio_recorder' not in st.session_state:
    st.session_state['audio_recorder'] = AudioRecorder()

# Recording buttons
from fix_audio import AudioRecorder  # See prior instructions or include class here

# Initialize recorder
if st.session_state.audio_recorder is None:
    st.session_state.audio_recorder = AudioRecorder()

# import soundfile as sf
# import numpy as np
# from scipy import signal
# import whisper

# … earlier code initializing st.session_state and AudioRecorder …

# 1) Recording buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ Start Recording"):
        st.session_state.recording = True
        st.session_state.audio_recorder.start_recording()
        st.success("Recording started!")

with col2:
    if st.button("⏹️ Stop & Transcribe"):
        st.session_state.recording = False
        # Stop and get raw numpy audio
        audio_data = st.session_state.audio_recorder.stop_recording()
        if audio_data is None:
            st.warning("No audio captured.")
        else:
            # Always overwrite the same temp file
            sf.write("temp_recording.wav", audio_data, 16000)
            st.session_state.transcript = ""  # clear previous
            # Preprocess & Transcribe from file
            model = whisper.load_model("base")
            # Bypass FFmpeg by loading via soundfile
            wav, sr = sf.read("temp_recording.wav")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = signal.resample(wav, int(len(wav) * 16000 / sr))
            wav = wav.astype(np.float32)
            result = model.transcribe(wav)
            st.session_state.transcript = result["text"]
            st.success("Transcription complete!")

# 2) Display Transcript
if st.session_state.transcript:
    st.subheader("Transcript")
    st.text_area("", st.session_state.transcript, height=200)


            # with st.spinner("Transcribing..."):
            #     model = whisper.load_model("base")
            #     result = model.transcribe('temp_recording.wav')
            #     st.session_state['transcript'] = result['text']
            # duplicate transcript display removed (placeholder replaced)st.text_area("Transcript:", st.session_state['transcript'], height=200)


    st.markdown('---')

    st.markdown('<div class="small-muted">Options</div>', unsafe_allow_html=True)
    use_diar = st.checkbox("Enable diarization", value=False, key='use_diar')
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    whisper_choice = st.selectbox("Select a Whisper model for audio-to-text conversion", options=['small', 'medium', 'large'], index=0)
    summ_choice = st.selectbox("Select a model for summarization", options=[
        'facebook/bart-large-cnn',
        't5-small',
        'google/pegasus-xsum',
        'groqai/groq-summarizer'
    ], index=0, key='summ_choice')

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("Clear", key='clear_btn'):
            for k in ['audio_chunks', 'live_segments', 'diarized_transcript', 'summary', 'status', 'summary_accuracy', 'metrics', 'models_used']:
                st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else None
            st.session_state['status'] = 'Idle'
            st.session_state['models_used'] = {}
            st.session_state['metrics'] = {}

    with b3:
        if st.button("Submit", key='submit_btn'):
            if st.session_state['audio_chunks']:
                final = os.path.join(tempfile.gettempdir(), f"final_{uuid.uuid4().hex}.wav")
                data = np.vstack(st.session_state['audio_chunks'])
                sf.write(final, data, SAMPLE_RATE, format='WAV')
                threading.Thread(target=backend_pipeline, args=(final, st.session_state.get('use_diar', False)), daemon=True).start()
            else:
                st.session_state['status'] = 'No audio available to submit'
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.markdown('<h4>Live transcript</h4>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">(Live updates while recording)</div>', unsafe_allow_html=True)
    live_placeholder = st.empty()
    # Add this block at the beginning of your app.py, after imports
if 'recording' not in st.session_state:
    st.session_state['recording'] = False
    pdf = create_pdf_bytes(pdf_text, title=meeting_title)
        # If we produced a real PDF (reportlab or fpdf), serve as application/pdf; otherwise fall back to plain text download.
    if REPORTLAB_AVAILABLE in (True, "FPDF"):
        st.download_button("Download PDF", pdf, f"{safe_filename(meeting_title)}.pdf", mime="application/pdf")
    else:
        st.download_button("Download (plain text fallback)", pdf, f"{safe_filename(meeting_title)}.txt", mime="text/plain")
    st.session_state['transcript'] = ""

if 'audio_buffer' not in st.session_state:
    st.session_state['audio_buffer'] = []

if 'live_transcript' not in st.session_state:
    st.session_state['live_transcript'] = ""

if 'audio_queue' not in st.session_state:
    st.session_state['audio_queue'] = []


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h4>Summary & Exports</h4>', unsafe_allow_html=True)
    if st.session_state.get('summary'):
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.write(st.session_state['summary'])
        st.markdown('</div>', unsafe_allow_html=True)
        acc = st.session_state.get('summary_accuracy')
        if acc is not None:
            st.markdown(f"**ROUGE-1 F1 (summary vs transcript):** {acc:.1f}%")
        meeting_title = st.text_input("Meeting title", value=st.session_state.get('meeting_title', 'Meeting'))
        meeting_date = st.text_input("Meeting date (ISO)", value=st.session_state.get('meeting_date', datetime.utcnow().isoformat()))
        md = f"# {meeting_title}\n\n**Date:** {meeting_date}\n\n## Summary\n\n{st.session_state['summary']}\n\n## Transcript\n"
        for s in st.session_state['diarized_transcript'] or []:
            md += f"- **{s['speaker']}** ({s['start']:.1f}-{s['end']:.1f}): {s['text']}\n"
        st.download_button("Download Markdown", md, f"{safe_filename(meeting_title)}.md")
        #  ...existing code...
        pdf_text = f"{meeting_title}\n\n{st.session_state['summary']}\n\nTranscript:\n"
        for s in st.session_state['diarized_transcript'] or []:
            pdf_text += f"{s['speaker']} ({s['start']:.1f}-{s['end']:.1f}): {s['text']}\n"
        
        pdf = create_pdf_bytes(pdf_text, title=meeting_title)
        
        # Fixed: Check if PDF library is available
        if REPORTLAB_AVAILABLE in (True, "FPDF"):
            st.download_button("Download PDF", pdf, f"{safe_filename(meeting_title)}.pdf", mime="application/pdf")
        else:
            st.download_button("Download (plain text fallback)", pdf, f"{safe_filename(meeting_title)}.txt", mime="text/plain")
        
        # Save Session button
        if st.button("Save Session (JSON + Parquet)"):
# // ...existing code...
            session_data = {
                "title": meeting_title,
                "date": meeting_date,
                "duration_seconds": st.session_state['metrics'].get('duration_seconds', compute_duration_from_segments(st.session_state.get('diarized_transcript', []))),
                "raw_transcript": " ".join([s['text'] for s in st.session_state.get('live_segments', [])]),
                "diarized_transcript": st.session_state.get('diarized_transcript'),
                "summary": st.session_state.get('summary'),
                "speakers_meta": [{"speaker": s['speaker'], "start": s['start'], "end": s['end']} for s in st.session_state.get('diarized_transcript', [])],
                "models": st.session_state.get('models_used', {}),
                "metrics": st.session_state.get('metrics', {}),
                "saved_at": datetime.utcnow().isoformat()
            }
            try:
                jpath = save_session_json(SESSIONS_DIR, session_data)
                ppath = save_session_parquet(SESSIONS_DIR, session_data)
                st.success(f"Saved session: {os.path.basename(jpath)} , {os.path.basename(ppath)}")
            except Exception as e:
                st.error(f"Save failed: {e}")
        email_to = st.text_input("Send email to (optional):", key="email_to")
        if st.button("Send Email") and email_to:
            subject = f"Meeting Summary - {meeting_title} - {meeting_date}"
            try:
                msg = EmailMessage()
                msg['Subject'] = subject
                smtp_cfg = st.secrets.get("smtp", {}) if hasattr(st, "secrets") else {}
                if smtp_cfg:
                    from_addr = smtp_cfg.get("from", smtp_cfg.get("user"))
                else:
                    from_addr = "you@example.com"
                msg['From'] = from_addr
                msg['To'] = email_to
                msg.set_content(md)
                msg.add_attachment(pdf, maintype='application', subtype='pdf', filename=f"{safe_filename(meeting_title)}.pdf")
                if smtp_cfg:
                    host = smtp_cfg.get("host")
                    port = smtp_cfg.get("port", 587)
                    user = smtp_cfg.get("user")
                    pwd = smtp_cfg.get("pass")
                    if not host or not user or not pwd:
                        raise Exception("SMTP config incomplete in secrets")
                    server = smtplib.SMTP(host, int(port))
                    server.starttls()
                    server.login(user, pwd)
                    server.send_message(msg)
                    server.quit()
                    st.success("Email sent via configured SMTP")
                else:
                    with smtplib.SMTP('localhost') as s:
                        s.send_message(msg)
                    st.success("Email sent via localhost SMTP")
            except Exception as e:
                st.error(f"Email failed: {e}")
    else:
        st.markdown('<div class="small-muted">Summary will appear here after processing audio.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.markdown('<h4>Saved Sessions</h4>', unsafe_allow_html=True)
    files = list_session_files(SESSIONS_DIR)
    if files:
        rows = []
        for f in files:
            rows.append({"file": os.path.basename(f), "path": f, "modified": datetime.fromtimestamp(os.path.getmtime(f)).isoformat()})
        df = pd.DataFrame(rows)
        st.dataframe(df[['file', 'modified']].rename(columns={"file": "File", "modified": "Modified"}), height=180)
        sel = st.selectbox("Select a saved file to view (JSON)", options=[r["path"] for r in rows])
        if sel:
            data = load_json_session(sel)
            if data:
                st.json(data)
            else:
                st.warning("Unable to load JSON (file may be parquet or corrupted).")
    else:
        st.markdown('<div class="small-muted">No saved sessions yet. Use "Save Session" after generating a summary.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')
st.info("Notes:\n- For diarization, accept access on Hugging Face and set `HF_TOKEN`.\n- For SMTP email, configure `st.secrets['smtp']` (host, port, user, pass, from).")
