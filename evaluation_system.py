
import os
import json
import time
import statistics
import requests
from datetime import datetime, timezone

# Set these as environment variables or paste keys directly here
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY",   "YOUR_GEMINI_API_KEY_HERE")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY",     "YOUR_GROQ_API_KEY_HERE")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")

PUN_METRICS = {
    "humor":     "How funny or humorous is this sentence? (1=not funny, 5=very funny)",
    "ambiguity": "Does the sentence use a word with double meaning? (1=no, 5=clear double meaning)",
    "coherence": "Is the sentence grammatically correct and clear? (1=confusing, 5=very clear)",
    "wordplay":  "How clever is the wordplay? (1=none, 5=very clever)",
}
LYRIC_METRICS = {
    "lyricism":  "How much does this sound like a song lyric? (1=not lyrical, 5=very lyrical)",
    "ambiguity": "Does the sentence use a word with double meaning? (1=no, 5=clear double meaning)",
    "emotion":   "Does the sentence evoke emotion or imagery? (1=flat, 5=very evocative)",
    "flow":      "Does the sentence have a natural rhythm for a song? (1=choppy, 5=smooth)",
}


# 
# PROMPT BUILDER
#

def _build_prompt(sentence, content_type):
    metrics = PUN_METRICS if content_type == "pun" else LYRIC_METRICS
    keys    = list(metrics.keys())
    metrics_block = "\n".join(f"  - {k}: {v}" for k, v in metrics.items())
    return f"""You are an expert evaluator of {'humor and puns' if content_type == 'pun' else 'song lyrics'}.
Evaluate this sentence on each metric using a score from 1 to 5.
Sentence: "{sentence}"
Metrics:
{metrics_block}
Return ONLY a valid JSON object with keys {json.dumps(keys)} and integer values 1-5. No extra text.
Example: {json.dumps({k: 3 for k in keys})}"""


# 
# API CALLERS
# 

def _call_gemini(prompt):
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("  [Gemini] Key not set – skipping.")
        return None
    try:
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return json.loads(text.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"  [Gemini] Error: {e}")
        return None

def _call_groq(prompt):
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("  [Groq] Key not set – skipping.")
        return None
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 200}
        resp    = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(text.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"  [Groq] Error: {e}")
        return None

def _call_openrouter(prompt):
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("  [OpenRouter] Key not set – skipping.")
        return None
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "openai/gpt-oss-120b:free", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 200}
        resp    = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(text.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"  [OpenRouter] Error: {e}")
        return None


# 
# SCORING
# 

def _aggregate(per_llm_scores):
    all_metrics = set()
    for s in per_llm_scores.values():
        if s: all_metrics.update(s.keys())
    if not all_metrics:
        return {"composite": 0.0}
    metric_avgs = {}
    for metric in all_metrics:
        vals = [s[metric] for s in per_llm_scores.values() if s and metric in s and isinstance(s[metric], (int, float))]
        metric_avgs[metric] = round(statistics.mean(vals), 2) if vals else 0.0
    return {**metric_avgs, "composite": round(statistics.mean(metric_avgs.values()), 2)}


# 
# MAIN INTERFACE
# 

def evaluate_candidates(candidates, output_file="evaluation_results.json"):
    """
    Evaluate a list of candidate sentences.

    Args:
        candidates:  list of {"text": <str>, "type": "pun" | "lyric"}
        output_file: where to save results as JSON

    Returns:
        List of result dicts sorted by composite score (highest first).
    """
    print(f"\nEvaluating {len(candidates)} candidate(s)...")
    results = []

    for i, cand in enumerate(candidates, 1):
        sentence     = cand.get("text", "")
        content_type = cand.get("type", "pun")
        print(f'\n[{i}/{len(candidates)}] ({content_type}): "{sentence[:70]}"')

        prompt   = _build_prompt(sentence, content_type)
        per_llm  = {
            "gemini":   _call_gemini(prompt),
            "groq":     _call_groq(prompt),
            "openrouter": _call_openrouter(prompt),
        }
        results.append({
            "sentence":  sentence,
            "type":      content_type,
            **per_llm,
            "aggregate": _aggregate({k: v for k, v in per_llm.items() if v}),
        })

    results.sort(key=lambda r: r["aggregate"].get("composite", 0), reverse=True)

    with open(output_file, "w") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "results": results}, fh, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    return results


def print_summary(results):
    print("\n" + "="*60)
    print("EVALUATION SUMMARY  (sorted by composite score)")
    print("="*60)
    for rank, r in enumerate(results, 1):
        comp = r["aggregate"].get("composite", 0)
        text = r["sentence"][:60] + ("…" if len(r["sentence"]) > 60 else "")
        print(f'  #{rank} [{r["type"].upper()}] score={comp:.2f}  "{text}"')
        for metric, val in r["aggregate"].items():
            if metric != "composite":
                print(f"       {metric}: {val}")


# 
# TEST  (replace with real sentences once Hyun Tae's module is ready)
# 
if __name__ == "__main__":
    dummy_candidates = [
        {"text": "Am I the only one who realizes that blackboards are truly remarkable?", "type": "pun"},
        {"text": "car go wroom wroom, i like",               "type": "lyric"},
        {"text": "I've been saving up these tears for a rainy day I'll never spend.",     "type": "lyric"},
    ]
    results = evaluate_candidates(dummy_candidates)
    print_summary(results)   