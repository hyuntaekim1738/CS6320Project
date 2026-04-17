# One Billion Word Dataset Setup

Required for TF-IDF and Word2Vec methods to produce better results. Without it the code falls back to the Brown corpus automatically.

---

## Download
1. Go to **statmt.org/lm-benchmark/**
2. Download `1-billion-word-language-modeling-benchmark-r13output.tar.gz`
3. Extract it

## Upload to Google Drive
Upload the extracted folder to your Google Drive.

## Mount Drive in Colab
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Set the path
```python
import os
os.environ["ONE_BILLION_WORD_PATH"] = "/content/drive/MyDrive/YOUR_FOLDER/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"
```
Replace `YOUR_FOLDER` with wherever you uploaded it in Drive.

Run this cell before running `context_word_generator.py`.
