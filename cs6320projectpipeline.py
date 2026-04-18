# -*- coding: utf-8 -*-
"""Python version of CS6320ProjectPipeline.ipynb

# CS6320 Project Pipeline
This contains the final pipeline that is used to run our experiment
## Required Files
- subtask3-homographic-trial.xml
- subtask3-homographic-trial.gold
- models/*
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install gensim

import os
import nltk
from nltk.corpus import wordnet as wn, stopwords, brown
import xml.etree.cElementTree as ET
import json
import re
import requests
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline, AutoModelForSequenceClassification, AutoTokenizer

nltk.download('wordnet')
nltk.download("stopwords", quiet=True)
nltk.download("brown", quiet=True)

os.environ["ONE_BILLION_WORD_PATH"] = "./models/1-billion-words/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"
# Set these as environment variables or paste keys directly here
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY",     "YOUR_GEMINI_API_KEY_HERE")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY",       "YOUR_GROQ_API_KEY_HERE")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
TOP_N = 10

"""## Retrieving Input Words"""
def parse_xml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    data = []
    for texts in root.findall("text"):
        for word in texts.findall("word"):
            if word.attrib.get("senses") == "2":
                data.append({
                    "word_id": word.attrib["id"],
                    "pun_word": word.text
                })
    return data

def parse_gold(filepath):
    gold = {}
    with open(filepath, "r") as f:
        for line in f:
            context = line.strip().split("\t")
            if len(context) < 3:
                continue  # skip malformed lines
            id = context[0]
            sense1 = context[1]
            sense2 = context[2].split(";")[0]
            gold[id] = [sense1, sense2]
    return gold

def get_def(pun):
    try:
        lem = wn.lemma_from_key(pun)
        return lem.synset().definition()
    except:
        return None

def build_test_set(xml, gold):
    xml_data = parse_xml(xml)
    gold_data = parse_gold(gold)

    test = []

    for r in xml_data:
        id = r["word_id"]
        puns = r['pun_word']

        if id not in gold_data:
            continue

        keys = gold_data[id]

        definitions = []
        for k in keys:
            define = get_def(k)
            if define:
                definitions.append(define)

        if len(definitions) >= 2:
            test.append({
                "pun_word": puns,
                "Definitions": definitions
            })

    return test

"""## Get Related Words"""

print("Creating embeddings for related word generation (estimated runtime of 5 minutes)")
model = SentenceTransformer("all-MiniLM-L6-v2")

candidates = list(set(
    lem.name().replace("_", " ")
    for syn in wn.all_synsets()
    for lem in syn.lemmas()
    if "_" not in lem.name()
))

candidate_embeddings = model.encode(candidates, convert_to_tensor=True).cpu()
candidate_embeddings = torch.tensor(candidate_embeddings)

def def_to_related(define, first_words=5):
    def_embedding = model.encode(define, convert_to_tensor=True).cpu()
    scores = util.cos_sim(def_embedding, candidate_embeddings.cpu())[0]
    top_indices = scores.topk(first_words).indices
    return [candidates[i] for i in top_indices]

def create_related(test):
    output = []

    for item in test:
      for enteries in item:
          pun = enteries["pun_word"]
          def1, def2 = enteries["Definitions"]

          rel1 = def_to_related(def1)
          rel2 = def_to_related(def2)

          output.append({
              "pun_word": pun,
              "related_1": rel1,
              "related_2": rel2
          })

    return output

"""## Get Context Words"""


#
# SHARED HELPERS
#

def _stop_words():
    return set(stopwords.words("english"))

def _rake_keywords(text, stop_words):
    splitter = re.compile(r"[^\w\s]|\b(?:" + "|".join(re.escape(w) for w in stop_words) + r")\b", re.IGNORECASE)
    phrases = [p.strip() for p in splitter.split(text) if p.strip()]
    freq, degree = Counter(), Counter()
    for phrase in phrases:
        words = phrase.split()
        for w in words:
            freq[w] += 1
            degree[w] += len(words) - 1
    scores = {w: (degree[w] + freq[w]) / max(freq[w], 1) for w in freq}
    return [w.lower() for w in sorted(scores, key=lambda w: scores[w], reverse=True) if w.isalpha()]

def _load_corpus(related_words, max_sentences=5000):
    related_lower = {w.lower() for w in related_words}
    sentences = []

    # use One Billion Word dataset if path is set
    obw_path = os.environ.get("ONE_BILLION_WORD_PATH", "")
    if obw_path and os.path.isfile(obw_path):
        with open(obw_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if any(rw in line.lower() for rw in related_lower):
                    sentences.append(line)
                if len(sentences) >= max_sentences:
                    break
        if sentences:
            return sentences
    # fallback to Brown corpus
    for sent in brown.sents():
        sent_str = " ".join(sent)
        if any(rw in sent_str.lower() for rw in related_lower):
            sentences.append(sent_str)
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        sentences = [f"The {w} is important in many contexts." for w in related_words]
    return sentences

#
# METHOD 1 – TF-IDF
#

def generate_tfidf_context(related_words_list):
    stop_words  = _stop_words()
    sentences   = _load_corpus(related_words_list)
    exclude     = {w.lower() for w in related_words_list} | stop_words

    keyword_docs = []
    for sent in sentences:
        kws = _rake_keywords(sent, stop_words)
        keyword_docs.append(" ".join(kws) if kws else sent)

    vec           = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix  = vec.fit_transform(keyword_docs)
    feature_names = vec.get_feature_names_out()
    scores        = tfidf_matrix.sum(axis=0).A1
    ranked        = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked if w not in exclude and w.isalpha()][:TOP_N]


#
# METHOD 2 – Word2Vec
#

def generate_word2vec_context(related_words_list, pun_word):
    exclude     = {w.lower() for w in related_words_list + [pun_word]} | _stop_words()
    corpus      = _load_corpus(related_words_list, max_sentences=10000)
    tokenized   = [s.lower().split() for s in corpus]
    model       = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
    wv          = model.wv

    candidate_scores = {}
    for rw in related_words_list:
        if rw.lower() not in wv:
            continue
        for word, score in wv.most_similar(rw.lower(), topn=20):
            if word.lower() not in exclude and word.isalpha():
                candidate_scores[word.lower()] = max(candidate_scores.get(word.lower(), 0.0), score)

    return sorted(candidate_scores, key=lambda w: candidate_scores[w], reverse=True)[:TOP_N]


#
# METHOD 3 – LLM (Gemini)
#

def generate_llm_context(related_words_list, pun_word, sense_name):
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print(f"[LLM] No Gemini API key set. Export GEMINI_API_KEY=<your_key> for {sense_name}.")
        return []

    prompt = f"""You are a linguist helping generate context words for a specific sense of a pun word.
Given a pun word, a specific sense, and related words for that sense, generate {TOP_N} context words
that are highly relevant to this specific sense.
Return ONLY a comma-separated list of single words, nothing else. Do NOT include the pun word or any of the related words.

Example:
Pun word: bark
Sense: Sense 1 (sound of a dog)
Related words for Sense 1: growl, howl, bite, woof
Context words: dog, animal, sound, loud, noisy, barks, puppy, listen, ear, hear

Now generate for:
Pun word: {pun_word}
Sense: {sense_name}
Related words for {sense_name}: {", ".join(related_words_list)}
Context words:"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    try:
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        exclude_words = {w.lower() for w in related_words_list + [pun_word]}
        return [w.strip().lower() for w in text.split(",") if w.strip().isalpha() and w.strip().lower() not in exclude_words][:TOP_N]
    except Exception as e:
        print(f"[LLM] Gemini error for {sense_name}: {e}")
        return []

# MAIN INTERFACE
def generate_context_words(related_words_sense1, related_words_sense2, pun_word, method="all"):
    results = {}
    methods = ["tfidf", "word2vec", "llm"] if method == "all" else [method]

    if "tfidf" in methods:
        print("[TF-IDF] Generating context words for Sense 1...")
        results["tfidf"] = {}
        results["tfidf"]["sense1"] = generate_tfidf_context(related_words_sense1)
        print(f"Sense 1: {results['tfidf']['sense1']}")

        print("[TF-IDF] Generating context words for Sense 2...")
        results["tfidf"]["sense2"] = generate_tfidf_context(related_words_sense2)
        print(f"Sense 2: {results['tfidf']['sense2']}")

    if "word2vec" in methods:
        print("[Word2Vec] Generating context words for Sense 1...")
        results["word2vec"] = {}
        results["word2vec"]["sense1"] = generate_word2vec_context(related_words_sense1, pun_word)
        print(f"Sense 1: {results['word2vec']['sense1']}")

        print("[Word2Vec] Generating context words for Sense 2...")
        results["word2vec"]["sense2"] = generate_word2vec_context(related_words_sense2, pun_word)
        print(f"Sense 2: {results['word2vec']['sense2']}")

    if "llm" in methods:
        print("[LLM] Generating context words via Gemini for Sense 1...")
        results["llm"] = {}
        results["llm"]["sense1"] = generate_llm_context(related_words_sense1, pun_word, "Sense 1")
        print(f"Sense 1: {results['llm']['sense1']}")

        print("[LLM] Generating context words via Gemini for Sense 2...")
        results["llm"]["sense2"] = generate_llm_context(related_words_sense2, pun_word, "Sense 2")
        print(f"Sense 2: {results['llm']['sense2']}")

    return results

"""### Generate Sentences"""

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def generate_sentence(pun, rela1, rela2, adjective):
    context = rela1[:2] + rela2[:2]

    prompt = (
        f"Create one clear, grammatically correct sentence using ALL these words: "
        f"{', '.join(context)}, {pun}. "
        f"Do not leave any word out."
    )
    #prompt = f"Create a funny sentence using: {', '.join(context)} and the word {pun}"

    input = tokenizer(prompt, return_tensors="pt")
    output = t5_model.generate(
        **input,
        max_length=50,
        num_return_sequences=3,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )

    sentences = [
        tokenizer.decode(out, skip_special_tokens=True)
        for out in output
    ]

    return sentences

def gen_all_sent(data, adjective="funny"):
    results = []

    for d in data:
        method = d["method"]
        pun = d["pun_word"]
        rela1 = d["related_1"]
        rela2 = d["related_2"]

        sentence = generate_sentence(pun, rela1, rela2, adjective)

        results.append({
            "method": method,
            "pun_word": pun,
            "sentences": sentence
        })

    return results

"""## LLM Alternative to Generating Sentences"""

def generate_llm_sentence(pun_word, context1, context2):
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print(f"[Groq] No Groq API key set.")
        return []

    prompt = f"""You are a linguist whose job it is to write clever puns and lyrics using double entendres.
Given a word with double meanings and context words for each definition of that word, generate a 2 sentences: one humorous
pun sentence that utilizes the double meaning of the word and one lyric sentence that utilizes the double meaning of the word. Both sentences
must contain the pun word and all context words. When returning, make the first sentence the pun and the second sentence the lyric. Both sentences
must be grammatically correct and clear. If context words are only provided for one of the definitions, use those context words for both sentences.
Return ONLY a semicolon separated list of these two sentences, nothing else.

Now generate for:
Pun word: {pun_word}
Context words for Sense 1: {', '.join(context1[:2])}
Context words for Sense 2: {', '.join(context2[:2])}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 200}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text.split(";")
    except Exception as e:
        print(f"[Groq] Error for pun word '{pun_word}': {e}")
        return []

# provides baseline for comparison
def generate_baseline(pun_word):
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print(f"[Groq] No Groq API key set.")
        return []

    prompt = f"""You are a linguist whose job it is to write clever puns and lyrics using double entendres.
Given a word with double meanings, generate 2 sentences: one humorous pun sentence that utilizes the double meaning of the word
and one lyric sentence that utilizes the double meaning of the word. Both sentences must contain the pun word.
When returning, make the first sentence the pun and the second sentence the lyric.
Both sentences
must be grammatically correct and clear. Return ONLY a semicolon separated list of these two sentences, nothing else.

Now generate for:
Pun word: {pun_word}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 200}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return [s.strip() for s in text.split(";")]
    except Exception as e:
        print(f"[Groq] Error for pun word '{pun_word}': {e}")
        return []

"""## Filter Sentences"""

def classify_text(candidates_with_metadata, model_dir):
  absolute_model_dir = os.path.abspath(model_dir)
  model = AutoModelForSequenceClassification.from_pretrained(absolute_model_dir, local_files_only=True)
  tokenizer = AutoTokenizer.from_pretrained(absolute_model_dir, local_files_only=True)

  classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
  )

  scored_candidates = []
  for cand in candidates_with_metadata:
    text = cand["text"]
    output = classifier(text)
    score_for_positive_label = next((s['score'] for s in output if s['label'] == 'LABEL_1'), 0.0)

    scored_candidates.append({
        "text": text,
        "score": score_for_positive_label,
        "method": cand["method"],
        "pun_word": cand["pun_word"]
    })
  ranked = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

  return ranked

import re

def normalize_lyric(text):
  text = text.lower()
  text = re.sub(r"[^\w\s]", "", text) # get rid of punctuation
  text = re.sub(r"\s+", " ", text).strip()
  return text

"""## Evaluation"""

import os
import time
import statistics
import requests
from datetime import datetime, timezone

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

"""## Format Helpers"""

def print_top_by_method(classified_list, content_type, top_n=3):
    print(f"\nTop {top_n} Classified {content_type}s by Method:")
    methods = set(item["method"] for item in classified_list)
    for method in methods:
        print(f"Method: {method.upper()}")
        method_specific_items = [item for item in classified_list if item["method"] == method]
        for i, item in enumerate(method_specific_items[:top_n]):
            print(f"{i+1}. {item['score']:.3f} {item['text']}")

def get_top_candidates_for_evaluation(classified_list, content_type, top_n=3):
    candidates_for_evaluation = []
    methods = set(item["method"] for item in classified_list)
    for method in methods:
        method_specific_items = [item for item in classified_list if item["method"] == method]
        # sort by score in descending order to make sure we pick the actual top items
        method_specific_items.sort(key=lambda x: x['score'], reverse=True)
        for item in method_specific_items[:top_n]:
            candidates_for_evaluation.append({
                "text": item["text"],
                "type": content_type
            })
    return candidates_for_evaluation

"""## Experiment Running"""
# T5 GENERATION APPROACH
# get input word
xml = "subtask3-homographic-trial.xml"
gold = "subtask3-homographic-trial.gold"
test = []
test.append(build_test_set(xml, gold))
print(f"Sample word {test[0][0]}")
# get related words
relate = create_related(test)
print(f"Related words {relate[0]}")
test_word = test[0][0]['pun_word']
related_1 = relate[0]['related_1']
related_2 = relate[0]['related_2']
print(related_1)
print(related_2)
# get context words
context = generate_context_words(
    related_words_sense1=related_1,
    related_words_sense2=related_2,
    pun_word=test_word,
    method="all"
)
print(f"CONTEXT: {context}")
# generate candidate sentences
print("\n\nRUNNING T5 GENERATION APPROACH")
data = []
for method_name, senses in context.items():
    d = {}
    d["method"] = method_name # Add the method name
    d["pun_word"] = test_word
    d["related_1"] = senses["sense1"]
    d["related_2"] = senses["sense2"]
    data.append(d)

all_generated_sentences_with_metadata = []
normalized_sentences = []
for trial in range(10):
    trial_results = gen_all_sent(data, "funny") # or "lyrical"
    for result_item in trial_results:
        method = result_item["method"]
        pun_word = result_item["pun_word"]
        for sentence_text in result_item["sentences"]:
            all_generated_sentences_with_metadata.append({
                "text": sentence_text,
                "method": method,
                "pun_word": pun_word
            })
            normalized_sentences.append({
                "text": normalize_lyric(sentence_text),
                "method": method,
                "pun_word": pun_word
            })
print(f"Generated {len(all_generated_sentences_with_metadata)} candidate sentences.")
# filter jokes and lyrics

lyrics_classified = classify_text(normalized_sentences, "./models/lyric")
jokes_classified = classify_text(all_generated_sentences_with_metadata, "./models/joke")

print("Top 5 Classified Lyrics:")
for i, l in enumerate(lyrics_classified[:5]):
    print(f"{l['score']:.3f} ({l['method']}) {l['text']}")

print("Top 5 Classified Jokes:")
for i, j in enumerate(jokes_classified[:5]):
    print(f"{j['score']:.3f} ({j['method']}) {j['text']}")
print_top_by_method(lyrics_classified, "Lyric", top_n=3)
print_top_by_method(jokes_classified, "Joke", top_n=3)
top_lyrics_for_eval = get_top_candidates_for_evaluation(lyrics_classified, "lyric", top_n=1)
top_jokes_for_eval = get_top_candidates_for_evaluation(jokes_classified, "pun", top_n=1)
baseline = generate_baseline(test_word)
print(f"Here is the baseline lyric and pun we'll be comparing to: {baseline}")

baseline_candidates = []
if len(baseline) > 0:
    baseline_candidates.append({"text": baseline[0], "type": "pun"})
if len(baseline) > 1:
    baseline_candidates.append({"text": baseline[1], "type": "lyric"})

all_candidates_for_evaluation = top_lyrics_for_eval + top_jokes_for_eval + baseline_candidates

print(f"Evaluating {len(all_candidates_for_evaluation)} selected candidates...")
results = evaluate_candidates(all_candidates_for_evaluation)
print_summary(results)

# LLM SENTENCE GENERATION APPROACH
print("\n\nRUNNING LLM SENTENCE GENERATION APPROACH")
all_generated_puns = []
all_generated_lyrics = []
for method_name, senses in context.items():
    pun_and_lyric = generate_llm_sentence(test_word, senses["sense1"], senses["sense2"])
    print(pun_and_lyric)
    d = {}
    d["method"] = method_name
    d["pun_word"] = test_word
    d["text"] = pun_and_lyric[0]
    all_generated_puns.append(d)
    if (len(pun_and_lyric) <= 1):
        print("Generation went faulty. Try again")
        exit()
    d = {}
    d["method"] = method_name
    d["pun_word"] = test_word
    d["text"] = normalize_lyric(pun_and_lyric[1])
    all_generated_lyrics.append(d)
# filter jokes and lyrics

lyrics_classified = classify_text(all_generated_lyrics, "./models/lyric")
jokes_classified = classify_text(all_generated_puns, "./models/joke")

print("Top 5 Classified Lyrics:")
for i, l in enumerate(lyrics_classified[:5]):
    print(f"{l['score']:.3f} ({l['method']}) {l['text']}")

print("Top 5 Classified Jokes:")
for i, j in enumerate(jokes_classified[:5]):
    print(f"{j['score']:.3f} ({j['method']}) {j['text']}")
print_top_by_method(lyrics_classified, "Lyric", top_n=3)
print_top_by_method(jokes_classified, "Joke", top_n=3)
top_lyrics_for_eval = get_top_candidates_for_evaluation(lyrics_classified, "lyric", top_n=1)
top_jokes_for_eval = get_top_candidates_for_evaluation(jokes_classified, "pun", top_n=1)
baseline = generate_baseline(test_word)
print(f"Here is the baseline lyric and pun we'll be comparing to: {baseline}")

baseline_candidates = []
if len(baseline) > 0:
    baseline_candidates.append({"text": baseline[0], "type": "pun"})
if len(baseline) > 1:
    baseline_candidates.append({"text": baseline[1], "type": "lyric"})

all_candidates_for_evaluation = top_lyrics_for_eval + top_jokes_for_eval + baseline_candidates

print(f"Evaluating {len(all_candidates_for_evaluation)} selected candidates...")
results = evaluate_candidates(all_candidates_for_evaluation)
print_summary(results)