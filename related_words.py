import json
import re
from nltk.corpus import wordnet as wn

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