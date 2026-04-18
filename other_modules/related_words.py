import json
import re
from nltk.corpus import wordnet as wn

import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
import os

def create_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    candidates = list(set(
        lem.name().replace("_", " ")
        for syn in wn.all_synsets()
        for lem in syn.lemmas()
        if "_" not in lem.name()
    ))

    candidate_embeddings = model.encode(candidates, convert_to_tensor=False)

    save_path = "/content/drive/MyDrive/CS6320Models/reverseDict/"
    os.makedirs(save_path, exist_ok=True)  # creates folder if it doesn't exist

    np.save(save_path + "candidates.npy", candidate_embeddings)
    with open(save_path + "candidate_words.txt", "w") as f:
        f.write("\n".join(candidates))

def def_to_related(define, first_words=5):
    words = set()

    tokens = re.findall(r'\b\w+\b', define.lower())

    for token in tokens:
        synset = wn.synsets(token)

        for syns in synset:
            for lem in syns.lemmas():
                words.add(lem.name().replace("_", " "))

    return list(words)[:first_words]

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
    
def save_related(test, path="data/related_words.json"):
    with open(path, "w") as f:
        json.dump(test, f, indent=2)

if __name__ == "__main__":
    with open("data/test_set.json") as f:
        test = json.load(f)

    relate = create_related(test)
    save_related(relate)