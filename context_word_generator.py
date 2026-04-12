
import os
import re
import requests
from collections import Counter

import nltk
from nltk.corpus import stopwords, brown
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

nltk.download("stopwords", quiet=True)
nltk.download("brown", quiet=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
TOP_N = 10


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

def generate_tfidf_context(related_words_sense1, related_words_sense2):
    all_related = related_words_sense1 + related_words_sense2
    stop_words  = _stop_words()
    sentences   = _load_corpus(all_related)
    exclude     = {w.lower() for w in all_related} | stop_words

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

def generate_word2vec_context(related_words_sense1, related_words_sense2, pun_word):
    all_related = related_words_sense1 + related_words_sense2
    exclude     = {w.lower() for w in all_related + [pun_word]} | _stop_words()
    corpus      = _load_corpus(all_related, max_sentences=10000)
    tokenized   = [s.lower().split() for s in corpus]
    model       = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
    wv          = model.wv

    candidate_scores = {}
    for rw in all_related:
        if rw.lower() not in wv:
            continue
        for word, score in wv.most_similar(rw.lower(), topn=20):
            if word.lower() not in exclude and word.isalpha():
                candidate_scores[word.lower()] = max(candidate_scores.get(word.lower(), 0.0), score)

    return sorted(candidate_scores, key=lambda w: candidate_scores[w], reverse=True)[:TOP_N]


# 
# METHOD 3 – LLM (Gemini)
# 

def generate_llm_context(related_words_sense1, related_words_sense2, pun_word):
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("[LLM] No Gemini API key set. Export GEMINI_API_KEY=<your_key>")
        return []

    prompt = f"""You are a linguist helping generate context words for pun sentences.
Given a pun word with two senses and related words for each sense, generate {TOP_N} context words
that naturally bridge both senses and could appear in a sentence with the pun word.
Return ONLY a comma-separated list of single words, nothing else. Do NOT include the pun word.

Example:
Pun word: bark
Sense 1 related words: growl, howl, bite, woof
Sense 2 related words: trunk, branch, wood, rough
Context words: rough, loud, outside, nature, sharp, sound, covered, texture, animal, familiar

Now generate for:
Pun word: {pun_word}
Sense 1 related words: {", ".join(related_words_sense1)}
Sense 2 related words: {", ".join(related_words_sense2)}
Context words:"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    try:
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return [w.strip().lower() for w in text.split(",") if w.strip().isalpha()][:TOP_N]
    except Exception as e:
        print(f"[LLM] Gemini error: {e}")
        return []


# 
# MAIN INTERFACE
# 

def generate_context_words(related_words_sense1, related_words_sense2, pun_word, method="all"):
    results = {}
    methods = ["tfidf", "word2vec", "llm"] if method == "all" else [method]

    if "tfidf" in methods:
        print("[TF-IDF] Generating context words...")
        results["tfidf"] = generate_tfidf_context(related_words_sense1, related_words_sense2)
        print(f"  → {results['tfidf']}")

    if "word2vec" in methods:
        print("[Word2Vec] Generating context words...")
        results["word2vec"] = generate_word2vec_context(related_words_sense1, related_words_sense2, pun_word)
        print(f"  → {results['word2vec']}")

    if "llm" in methods:
        print("[LLM] Generating context words via Gemini...")
        results["llm"] = generate_llm_context(related_words_sense1, related_words_sense2, pun_word)
        print(f"  → {results['llm']}")

    return results


# 
# TEST
# 

if __name__ == "__main__":
    context = generate_context_words(
        related_words_sense1=["money", "vault", "loan", "finance", "savings"],
        related_words_sense2=["shore", "river", "mud", "stream", "slope"],
        pun_word="bank",
        method="all"
    )
    print("\n── Results ──")
    for method_name, words in context.items():
        print(f"  {method_name:10s}: {words}")