# CS6320 Project
## Results
To refer to the specific set of results we referenced in our report, go to the Copy_of_CS6320ProjectPipeline.ipynb notebook. 
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
The output will be displayed on the terminal and saved to evaluation_results.json.

The models, other_modules, and pun_lyric_filtering directories contain the functions and modules that were used in the cs6320projectpipeline.py file. The semeval2017_task7 contains the dataset containing the input set of words.

## Utilized Datasets
- SemEval 2017 Task 7: https://alt.qcri.org/semeval2017/task7/
- ColBERT: https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection
- Million Song Dataset: https://huggingface.co/datasets/vishnupriyavr/spotify-million-song-dataset
- One Million Reddit Questions Dataset: https://huggingface.co/datasets/SocialGrep/one-million-reddit-questions
- TLDR-17: https://huggingface.co/datasets/webis/tldr-17
- One Billion Word Dataset: https://www.statmt.org/lm-benchmark/
- Brown Corpus (Included in NLTK)
