
"""
Evaluate summaries with ROUGE and BLEU.
Requires reference summaries in folder: references/{sample_id}.ref.txt
Generated summaries should be in outputs/{sample_id}.gen.txt
"""

from rouge_score import rouge_scorer
import sacrebleu
import glob, os

def score_pair(reference_text, generated_text):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    # BLEU via sacrebleu
    bleu = sacrebleu.sentence_bleu(generated_text, [reference_text]).score
    return {"rouge1": scores["rouge1"].fmeasure, "rouge2": scores["rouge2"].fmeasure, "rougeL": scores["rougeL"].fmeasure, "bleu": bleu}

def evaluate_folder(ref_folder="references", gen_folder="outputs"):
    pairs = []
    for ref_path in glob.glob(os.path.join(ref_folder, "*.ref.txt")):
        base = os.path.basename(ref_path).replace(".ref.txt", "")
        gen_path = os.path.join(gen_folder, base + ".gen.txt")
        if not os.path.exists(gen_path):
            print("Missing generated:", gen_path)
            continue
        with open(ref_path, "r", encoding="utf-8") as f:
            ref = f.read().strip()
        with open(gen_path, "r", encoding="utf-8") as f:
            gen = f.read().strip()
        s = score_pair(ref, gen)
        print(f"{base} -> ROUGE1: {s['rouge1']:.3f}, ROUGE2: {s['rouge2']:.3f}, ROUGE-L: {s['rougeL']:.3f}, BLEU: {s['bleu']:.2f}")
