import numpy as np
from tqdm import tqdm


def merge_transcriptions(diarization_txt_path, transcript_segments, diarize_df):
    # If True, assign speakers even when there's no direct time overlap
    fill_nearest = True

    with open(diarization_txt_path, "w", encoding="utf-8") as f:
        for seg in tqdm(transcript_segments, desc="🔗 Merging speakers with transcript", unit="segment"):
            # assign speaker to segment (if any)
            diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
            diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
            # remove no hit, otherwise we look for closest (even negative intersection...)
            if not fill_nearest:
                dia_tmp = diarize_df[diarize_df['intersection'] > 0]
            else:
                dia_tmp = diarize_df
            if len(dia_tmp) > 0:
                # sum over speakers
                speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            else:
                speaker = "Unknown"

            seg["speaker"] = speaker

            f.write(f"[{speaker}] : {seg['text'].strip()}\n")

    return True

