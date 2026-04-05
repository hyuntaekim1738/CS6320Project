#script to make a lyrics vs. non-lyrics dataset like the ColBERT from the paper
# positive class: spotify million song dataset (huggingface and cco)
#negative class: reddit posts from huggingface datasets
#output: dataset.csv [text, label]  (1=lyrics, 0=nonlyric)
import re
import random
import csv
from datasets import load_dataset

TARGET_PER_CLASS = 100_000
MIN_TOKENS = 8 # min and max length
MAX_TOKENS = 80
OUTPUT_PATH = "dataset.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

#helper functions
def token_count(text):
    return len(text.split())

def within_length(text):
    return MIN_TOKENS <= token_count(text) <= MAX_TOKENS

def normalize(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text[0].upper() + text[1:] if text else ""

def clean_lyrics(raw):
    raw = re.sub(r"\[.*?\]|\(.*?\)", "", raw)
    raw = re.sub(r"\d+Embed$", "", raw.strip())

    stanzas = [s.strip() for s in re.split(r"\n{2,}", raw) if s.strip()]

    # if stanza splitting doesn't work, just group every 2 lines together
    if not any(within_length(normalize(s)) for s in stanzas):
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        stanzas = [" ".join(lines[i:i+2]) for i in range(0, len(lines), 2)]

    return [normalize(s) for s in stanzas if within_length(normalize(s))]

# getting positive examples
def load_lyrics():
    samples = []

    ds = load_dataset("vishnupriyavr/spotify-million-song-dataset", split="train", trust_remote_code=True)
    rows = list(ds)
    random.shuffle(rows)

    for row in rows:
        if len(samples) >= TARGET_PER_CLASS * 2:
            break
        raw = row.get("lyrics") or row.get("text") or row.get("Lyric", "")
        if not raw or not isinstance(raw, str):
            continue
        for chunk in clean_lyrics(raw):
            if len(samples) >= TARGET_PER_CLASS * 2:
                break
            samples.append(chunk)
    return samples

# loading negative examples
def load_reddit():
    samples = []

    try:
        ds1 = load_dataset("SocialGrep/one-million-reddit-questions", split="train", streaming=True, trust_remote_code=True)
        for row in ds1:
            if len(samples) >= TARGET_PER_CLASS // 2:
                break
            text = normalize(str(row.get("title") or row.get("content", "")))
            if within_length(text):
                samples.append(text)
    except Exception as e:
        print(f"source 1 failed: {e}")

    try:
        ds2 = load_dataset("webis/tldr-17", split="train", streaming=True, trust_remote_code=True)
        for row in ds2:
            if len(samples) >= TARGET_PER_CLASS * 2:
                break
            for field in ("summary", "content", "body", "normalizedBody"):
                raw = row.get(field, "")
                if not raw:
                    continue
                sentence = normalize(re.split(r"(?<=[.!?])\s+", raw.strip())[0])
                if within_length(sentence):
                    samples.append(sentence)
                    break
    except Exception as e:
        print(f"source 2 failed: {e}")

    return samples

# write to csv
def build_dataset():
    positives = load_lyrics()
    negatives = load_reddit()

    #combine, shuffle, dedup, trim
    pos_rows = list({t: None for t in positives}.keys())# dedup within class
    neg_rows = list({t: None for t in negatives}.keys())
    pos_rows = random.sample(pos_rows, min(len(pos_rows), TARGET_PER_CLASS))
    neg_rows = random.sample(neg_rows, min(len(neg_rows), TARGET_PER_CLASS))

    rows = [(t, 1) for t in pos_rows] + [(t, 0) for t in neg_rows]
    random.shuffle(rows)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"File has been saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_dataset()