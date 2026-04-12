from nltk.corpus import wordnet as wn
import xml.etree.cElementTree as ET
import json


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

def save_test_set(test, path="data/test_set.json"):
    with open(path, "w") as f:
        json.dump(test, f, indent=2)



if __name__ == "__main__":
    import nltk
    nltk.download('wordnet')

    xml = "subtask3-heterographic-trial.xml"
    gold = "subtask3-heterographic-trial.gold"

    test = []

    test.append(build_test_set(xml, gold))

    xml = "subtask3-homographic-trial.xml"
    gold = "subtask3-homographic-trial.gold"

    test.append(build_test_set(xml, gold))

    save_test_set(test)