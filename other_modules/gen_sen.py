import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def generate_sentence(pun, rela1, rela2):
    context = rela1[:2] + rela2[:2]

    prompt = f"Generate a sentence using the words: " + " ".join(context) + " and " + pun
    #prompt = f"Create a funny sentence using: {', '.join(context)} and the word {pun}"

    input = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
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

def gen_all_sent(data):
    results = []

    for d in data:
        pun = d["pun_word"]
        rela1 = d["related_1"]
        rela2 = d["related_2"]

        sentence = generate_sentence(pun, rela1, rela2)

        results.append({
            "pun_word": pun,
            "sentences": sentence
        })

    return results

def save_gen_sent(data, filepath="data/generate_sentences.json"):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

if __name__=="__main__":
    with open("data/related_words.json") as f:
        related = json.load(f)

    sentence = gen_all_sent(related)
    save_gen_sent(sentence)
