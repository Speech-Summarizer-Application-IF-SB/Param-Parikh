# prompts.py
"""
Prompt templates for meeting summarization.
Each template is a format string where we pass:
 - diarized_text: the transcript text with speaker labels/time ranges
 - meeting_type: e.g., "standup", "project sync", "client call"
 - length_hint: "short", "detailed", "bullet points", etc.
"""

SIMPLE_SUMMARY_PROMPT = """You are a meeting summarizer.
Meeting type: {meeting_type}
Length: {length_hint}

Below is a diarized transcript of a meeting (speaker labels included).
Write a concise summary (3-6 sentences) capturing key decisions, action items, and owners.
Preserve speaker attribution when needed. Use bullet points for action items.

Transcript:
{diarized_text}

Summary:"""

SPEAKER_WISE_PROMPT = """You are a helpful assistant. Produce:
1) A short summary per speaker (1-2 sentences each).
2) A consolidated meeting summary (3-5 sentences).
3) A list of action items in bullet points, with owner if mentioned.

Meeting type: {meeting_type}
Transcript:
{diarized_text}

Output format:
[SPEAKER_00] <one-line summary>
[SPEAKER_01] <one-line summary>

Consolidated Summary:
- <line>

Action Items:
- <Owner> : <action>
"""

BULLET_PROMPT = """Summarize the transcript into bullets grouped by topic.
Meeting Type: {meeting_type}
Length: {length_hint}

Transcript:
{diarized_text}

Bulleted summary:"""

def build_diarized_text(segments, max_segments=None):
    """
    Build a single string that contains the diarized transcript.
    segments: list of {"speaker","start","end","text"}
    max_segments: if provided, limit number of segments (for long meetings).
    """
    lines = []
    for i, s in enumerate(segments):
        if max_segments is not None and i >= max_segments:
            break
        # include times optionally
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        text = s.get("text", "").strip()
        speaker = s.get("speaker", f"SPEAKER_{i:02d}")
        lines.append(f"[{speaker}] ({start:.1f}-{end:.1f}s): {text}")
    return "\n".join(lines)
