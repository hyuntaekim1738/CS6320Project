# CS6320 Project
## Running Instructions
### Install dependencies
```bash
pip install sentencepiece sentence-transformers transformers scikit-learn numpy torch requests nltk scikit-learn gensim
```
### Setup API Keys
Get free keys (no credit card) from:
- **Gemini**: https://aistudio.google.com/
- **Groq**: https://console.groq.com/
- **OpenRouter**: https://openrouter.ai/

Gemini
1. Sign in with your Google account
2. Click "Get API key" in the top left
3. Click "Create API key"
4. Copy it

Groq
1. Click "Sign Up" — you can use your Google account
2. Once logged in, click "API Keys" in the left sidebar
3. Click "Create API Key"
4. Copy it immediately — it only shows once

OpenRouter
1. Click "Sign Up" — Google account works here too
2. Once logged in, click "Keys" in the left sidebar
3. Click "Create Key"
4. Copy it

Once you have your keys, navigate to CS6320Project/cs6320projectpipeline.py and paste them at the top of each file where it says:
```python
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY",     "YOUR_GEMINI_API_KEY_HERE")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY",       "YOUR_GROQ_API_KEY_HERE")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
```
Replace the `YOUR_X_API_KEY_HERE` placeholder with your actual key.

---
### Run the program
Ensure that the following files and folders are present:
- CS6320Project/subtask3-homographic-trial.gold
- CS6320Project/subtask3-homographic-trial.xml
- CS6320Project/models
- CS6320Project/cs6320projectpipeline.py.  
To run the entire system, run the following command:
```bash
python3 cs6320projectpipeline.py
```
The output will be displayed on the terminal

## Files
- `context_word_generator.py` — generates context words (TF-IDF, Word2Vec, LLM)
- `evaluation_system.py` — scores puns/lyrics using 3 LLM APIs

---


---


## Run tests
```bash
python context_word_generator.py   # TF-IDF and Word2Vec work without API keys
python evaluation_system.py        # needs all 3 keys, uses dummy sentences for now
```

---

## Pipeline usage

### context_word_generator.py
Takes output from `related_words.py`:
```python
from context_word_generator import generate_context_words

context = generate_context_words(
    related_words_sense1=["money", "vault", "loan"],
    related_words_sense2=["shore", "river", "stream"],
    pun_word="bank",
    method="all"  # or "tfidf" / "word2vec" / "llm"
)
# Returns: {"tfidf": [...], "word2vec": [...], "llm": [...]}
```

### evaluation_system.py
Takes output from pun/lyric filters:
```python
from evaluation_system import evaluate_candidates, print_summary

results = evaluate_candidates([
    {"text": "I tried to save money at the river bank...", "type": "pun"},
    {"text": "She banks on love, but shores up her heart.", "type": "lyric"},
])
print_summary(results)
# Results also saved to evaluation_results.json
```

**Pun metrics:** humor, ambiguity, coherence, wordplay (1–5 each)  
**Lyric metrics:** lyricism, ambiguity, emotion, flow (1–5 each)
