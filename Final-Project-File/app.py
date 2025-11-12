import os
import io
import time
import tempfile
import streamlit as st
from pydub import AudioSegment
from dotenv import load_dotenv
from main import (
    step_clean_audio,
    step_transcription,
    step_diarization,
    step_merge_transcripts,
    step_summarization,
)


# Load .env from project root before importing main
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
print("[ENV][app] PYANNOTE_API_KEY loaded:", bool(os.getenv("PYANNOTE_API_KEY")))

# === Function to process the full pipeline and return results ===
def process_pipeline(input_audio_bytes, status_placeholder):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            cleaned_audio = os.path.join(tmpdir, "cleaned.wav")
            transcript_txt = os.path.join(tmpdir, "transcript.txt")
            transcript_json = os.path.join(tmpdir, "transcription.json")
            diarization_json = os.path.join(tmpdir, "diarization.json")
            diarized_txt = os.path.join(tmpdir, "diarized_transcript.txt")
            summary_txt = os.path.join(tmpdir, "summary.txt")

            audio = AudioSegment.from_file(input_audio_bytes, format="wav")
            audio.export(input_path, format="wav")

            # Step 1
            st.session_state.status = "🔊 Cleaning audio..."
            status_placeholder.info(f"**Status:** {st.session_state.status}")
            with st.spinner("Cleaning audio... ⏳"):
                if not step_clean_audio(input_path, cleaned_audio):
                    return "Audio cleaning failed!", "", ""

            # Step 2
            st.session_state.status = "📝 Transcribing..."
            status_placeholder.info(f"**Status:** {st.session_state.status}")
            with st.spinner("Transcribing... 📝"):
                if not step_transcription(cleaned_audio, transcript_txt, transcript_json):
                    return "Transcription failed!", "", ""
                with open(transcript_txt, "r", encoding="utf-8") as f:
                    transcription = f.read()
                st.session_state.transcription = transcription

            # Step 3
            st.session_state.status = "👥 Performing diarization..."
            status_placeholder.info(f"**Status:** {st.session_state.status}")
            with st.spinner("Performing diarization... 👥"):
                if not step_diarization(cleaned_audio, diarization_json):
                    return "Diarization failed!", "", ""

            # Step 4
            st.session_state.status = "🔗 Merging results..."
            status_placeholder.info(f"**Status:** {st.session_state.status}")
            with st.spinner("Merging results... 🔗"):
                if not step_merge_transcripts(transcript_json, diarization_json, diarized_txt):
                    return "Merging failed!", "", ""
                with open(diarized_txt, "r", encoding="utf-8") as f:
                    diarized = f.read()
                st.session_state.diarized = diarized

            # Step 5
            st.session_state.status = "🧠 Summarizing..."
            status_placeholder.info(f"**Status:** {st.session_state.status}")
            with st.spinner("Summarizing... 🧠"):
                if not step_summarization(diarized_txt, summary_txt):
                    return "Summarization failed!", "", ""
                with open(summary_txt, "r", encoding="utf-8") as f:
                    summary = f.read()
                st.session_state.summary = summary

            st.session_state.status = "✅ Completed"
            status_placeholder.success(f"**Status:** {st.session_state.status}")

            return transcription, diarized, summary

    except Exception as e:
        return f"❌ Error: {e}", "", ""


# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Meeting Summarizer",
    layout="wide",
    page_icon="🎙️"
)

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    .main-title { 
        text-align: center; 
        font-size: 42px; 
        font-weight: 700; 
        color: #1f1f1f;
        margin-bottom: 10px;
    }
    .subtext { 
        text-align: center; 
        color: #666; 
        font-size: 16px;
        margin-bottom: 3rem; 
    }
    .section-box {
        background-color: white;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 20px;
    }
    .status-badge {
        background-color: #f0f0f0;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 20px;
    }
    /* Style for file uploader */
    .uploadedFile {
        border: 2px dashed #d0d0d0;
        border-radius: 8px;
        padding: 40px;
        text-align: center;
        background-color: #fafafa;
    }
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #4285f4;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        background-color: #3367d6;
    }
    /* Result boxes */
    .result-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        min-height: 300px;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
    }
    /* Audio player styling */
    audio {
        width: 100%;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- SESSION STATE -------------------
if "status" not in st.session_state:
    st.session_state.status = "Idle"
if "transcription" not in st.session_state:
    st.session_state.transcription = "" 
if "diarized" not in st.session_state:
    st.session_state.diarized = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "meeting_title" not in st.session_state:
    st.session_state.meeting_title = "Meeting"

# ------------------- PAGE HEADER -------------------
st.markdown("<h1 class='main-title'>Meeting Summarizer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtext'>Upload an audio file or record live. Get diarized transcript, concise summary, export & storage.</p>",
    unsafe_allow_html=True,
)

# Status display
st.markdown(f"<div class='status-badge'>Status: {st.session_state.status}</div>", unsafe_allow_html=True)

# ------------------- MAIN LAYOUT -------------------
col_left, col_right = st.columns([1, 1], gap="large")

# ------------------- LEFT COLUMN - AUDIO SECTION -------------------
with col_left:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>Audio</h3>", unsafe_allow_html=True)
    
    # Meeting title input
    st.session_state.meeting_title = st.text_input(
        "Meeting Title", 
        value=st.session_state.meeting_title,
        placeholder="Enter meeting title..."
    )
    
    st.markdown("---")
    
    # Input mode selection
    input_mode = st.radio(
        "**Choose Input Method:**",
        ["📁 Upload an audio file", "🎙️ Live Recording"],
        label_visibility="visible"
    )
    
    input_audio = None
    
    if input_mode == "📁 Upload an audio file":
        st.markdown("""
        <div style='text-align: center; padding: 20px; border: 2px dashed #d0d0d0; border-radius: 8px; background-color: #fafafa;'>
            <p style='color: #666;'>📤 Drag and drop file here</p>
            <p style='color: #999; font-size: 14px;'>Limit 200MB per file • WAV, MP3, M4A</p>
        </div>
        """, unsafe_allow_html=True)
        
        input_audio = st.file_uploader(
            "Upload audio file", 
            type=["wav", "mp3", "m4a"],
            label_visibility="collapsed"
        )
        
        if input_audio is not None:
            file_name = input_audio.name.lower()
            
            # Convert MP3 to WAV if needed
            if file_name.endswith(".mp3") or file_name.endswith(".m4a"):
                audio = AudioSegment.from_file(input_audio)
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                input_audio = wav_buffer
            
            st.audio(input_audio)
    
    elif input_mode == "🎙️ Live Recording":
        st.info("💾 Click the microphone button to start recording. Save the file to use it for transcription.")
        input_audio = st.audio_input("Record audio", sample_rate=48000)
        
        if input_audio is not None:
            st.audio(input_audio)
    
    # Process button
    if input_audio:
        # Check audio duration
        try:
            audio_segment = AudioSegment.from_file(input_audio, format="wav")
            duration_seconds = len(audio_segment) / 1000
            duration_minutes = duration_seconds / 60
            
            st.info(f"✅ Audio duration: {duration_minutes:.2f} minutes")
            
            if duration_minutes < 1:
                st.error("❌ The audio file must be at least **1 minute long**.")
            else:
                # Transcribe button
                if st.button("🔍 Transcribe", use_container_width=True):
                    st.session_state.status = "Processing..."
                    status_placeholder = st.empty()
                    
                    # Run pipeline
                    transcription, diarized, summary = process_pipeline(input_audio, status_placeholder)
                    
                    if transcription.startswith("❌") or "failed" in transcription:
                        st.error(transcription)
                    else:
                        st.success("✅ Processing completed!")
                        st.balloons()
                        
        except Exception as e:
            st.error(f"⚠️ Could not read audio file: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- RIGHT COLUMN - SUMMARY & EXPORTS -------------------
with col_right:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>Summary & Exports</h3>", unsafe_allow_html=True)
    
    if st.session_state.summary and st.session_state.summary != "":
        # Display summary
        st.markdown("**Meeting Summary:**")
        with st.container():
            st.markdown(f"<div class='result-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📥 Transcript",
                data=st.session_state.transcription,
                file_name=f"{st.session_state.meeting_title}_transcript.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📥 Diarized",
                data=st.session_state.diarized,
                file_name=f"{st.session_state.meeting_title}_diarized.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col3:
            st.download_button(
                "📥 Summary",
                data=st.session_state.summary,
                file_name=f"{st.session_state.meeting_title}_summary.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("Summary will appear here after processing audio.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Saved Sessions Section
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>Saved Sessions</h3>", unsafe_allow_html=True)
    st.info('No saved sessions yet. Use "Save Session" after generating a summary.')
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- FULL WIDTH RESULTS SECTION -------------------
if st.session_state.transcription and st.session_state.transcription != "":
    st.markdown("---")
    st.markdown("### 📋 Detailed Results")
    
    tab1, tab2, tab3 = st.tabs(["📝 Raw Transcript", "👥 Diarized Transcript", "🧠 Summary"])
    
    with tab1:
        st.markdown(f"<div class='result-box'>{st.session_state.transcription}</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f"<div class='result-box'>{st.session_state.diarized}</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown(f"<div class='result-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:#999;'>Built with ❤️ for AI_LIVE_MEETING_SUMMARIZER</p>", unsafe_allow_html=True)
