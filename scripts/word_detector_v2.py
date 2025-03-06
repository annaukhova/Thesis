import re
import pandas as pd
from gensim.models import KeyedVectors
import argparse
import os
import numpy as np
import spacy
import nltk
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, TFBertForTokenClassification
import tensorflow as tf

# Load lighter models at the top
word2vec = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin", binary=True)
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dictionary
anachronism_dict = {
    "telegram": {"year": 1837, "style": "formal", "era": "20th"},
    "phonograph": {"year": 1877, "style": "neutral", "era": "20th"},
    "telegraph": {"year": 1830, "style": "formal", "era": "20th"},
    "selfie": {"year": 2002, "style": "casual", "era": "21st"},
    "cringe": {"year": 2000, "style": "casual", "era": "21st"},
    "lit": {"year": 2000, "style": "slang", "era": "21st"},
    "smartphone": {"year": 1999, "style": "neutral", "era": "21st"},
    "internet": {"year": 1969, "style": "neutral", "era": "20th"},
    "digital": {"year": 1940, "style": "neutral", "era": "20th"},
    "social": {"year": 2004, "style": "casual", "era": "21st"},
    "likes": {"year": 2004, "style": "casual", "era": "21st"},
    "comments": {"year": 2004, "style": "casual", "era": "21st"},
    "cool": {"year": 1930, "style": "slang", "era": "20th"},
    "dope": {"year": 1980, "style": "slang", "era": "21st"},
    "quill": {"year": 1200, "style": "formal", "era": "pre-20th"},
    "carriage": {"year": 1500, "style": "formal", "era": "pre-20th"},
    "gramophone": {"year": 1887, "style": "neutral", "era": "20th"},
    "frock": {"year": 1400, "style": "formal", "era": "pre-20th"},
    "telephony": {"year": 1876, "style": "formal", "era": "20th"},
    "emoji": {"year": 1997, "style": "casual", "era": "21st"},
    "hashtag": {"year": 2007, "style": "casual", "era": "21st"},
    "vlog": {"year": 2000, "style": "casual", "era": "21st"},
    "swipe": {"year": 2010, "style": "casual", "era": "21st"},
    "stan": {"year": 2000, "style": "slang", "era": "21st"},
    "groovy": {"year": 1960, "style": "slang", "era": "20th"},
    "jive": {"year": 1920, "style": "slang", "era": "20th"},
    "rad": {"year": 1980, "style": "slang", "era": "20th"}
}

# Training data
train_texts = [
    "the telegram arrived swiftly",                    # 21st
    "she took a selfie",                              # 21st
    "the telegram arrived swiftly soldiers said it was lit",  # 20th, formal
    "she took a selfie feeling cringe then sent a telegram",  # 20th, casual
    "soldiers communicated through the wires",        # 20th
    "the phonograph played loudly",                   # 20th
    "she posted on social media",                     # 21st
    "the smartphone buzzed in 1920",                  # 20th
    "he sent a telegram feeling lit",                 # 21st
    "the battle was lit and soldiers cheered",        # 20th, formal
    "she said cool in 1910",                          # 20th, formal
    "he used a smartphone yesterday",                 # 21st
    "the telegraph hummed all day",                   # 20th
    "the war was lit in 1940",                        # 20th, formal
    "she took a selfie in 1900",                      # 20th
    "the general said it was dope",                   # 20th, formal
    "he posted a selfie online",                      # 21st
    "the phonograph was cool and loud",               # 20th, formal
    "the war was dope in 1918",                       # 20th, formal
    "she said it was lit today",                      # 21st, casual
    "the telegram buzzed with news",                  # 21st
    "soldiers called it cool in 1945",                # 20th, formal
    "he took a selfie with a phonograph",             # 20th
    "she sent a telegram in 2023",                    # 21st
    "the war ended with a cool victory",              # 20th, formal
    "he used a telegraph in 2025",                    # 21st
    "the news came via telegram now",                 # 21st
    "soldiers sent telegrams in 2020",                # 21st
    "he wrote with a quill in 2023",                  # 21st
    "she sent a telegram yesterday",                  # 21st
    "the news came via telegraph now",                # 21st
    "they used a gramophone today",                   # 21st
    "her frock shone in 2024",                        # 21st
    "telephony ruled their chat",                     # 21st
    "the carriage rolled into town",                  # 21st
    "he wrote a letter with a quill",                 # 21st
    "telegraph lines buzzed in 2025",                 # 21st
    "she wore a frock to the club",                   # 21st
    "the war was dope in 1919",                       # 20th, formal
    "soldiers called it lit in 1945",                 # 20th, formal
    "the battle felt stan worthy",                    # 20th, formal
    "he said it was rad in 1930",                     # 20th, formal
    "the telegram was totally dope",                  # 20th, formal
    "their victory was lit in 1920",                  # 20th, formal
    "she called it cool in 1910",                     # 20th, formal
    "the phonograph was groovy",                      # 20th, formal
    "he spoke in jive at headquarters",               # 20th, formal
    "soldiers wrote letters home",                    # 20th
    "the radio played softly",                        # 20th
    "she read the news daily",                        # 20th
    "he traveled by train",                           # 20th
    "they met at the station",                        # 20th
    "the book was on the table",                      # 20th
    "she wore a hat to town",                         # 20th
    "the telegram arrived in 1940",                   # 20th, OK
    "soldiers used telegrams in 1918",                # 20th, OK
    "he sent a telegraph during the war"              # 20th, OK
]
train_labels = [
    [0, 1, 0, 0],                  # "telegram" = ANACHRONISM
    [0, 0, 0, 0],                  # All OK
    [0, 0, 0, 0, 0, 0, 0, 0, 2],  # "lit" = STYLE_MISMATCH
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # "selfie", "cringe" = ANACHRONISM
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 0, 0],                  # All OK
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 1, 0, 0],               # "smartphone" = ANACHRONISM
    [0, 0, 0, 1, 0, 2],            # "telegram" = ANACHRONISM, "lit" = STYLE_MISMATCH
    [0, 0, 0, 2, 0, 0, 0],         # "lit" = STYLE_MISMATCH
    [0, 0, 0, 2, 0],               # "cool" = STYLE_MISMATCH
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 0, 2, 0],               # "lit" = STYLE_MISMATCH
    [0, 0, 0, 1, 0],               # "selfie" = ANACHRONISM
    [0, 0, 0, 0, 2],               # "dope" = STYLE_MISMATCH
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 0, 2, 0],               # "cool" = STYLE_MISMATCH
    [0, 0, 0, 2, 0],               # "dope" = STYLE_MISMATCH
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 1, 0, 0],               # "telegram" = ANACHRONISM
    [0, 0, 0, 0, 2],               # "cool" = STYLE_MISMATCH
    [0, 0, 0, 1, 0, 0],            # "selfie" = ANACHRONISM
    [0, 0, 0, 1, 0],               # "telegram" = ANACHRONISM
    [0, 0, 0, 0, 2, 0],            # "cool" = STYLE_MISMATCH
    [0, 0, 0, 1, 0],               # "telegraph" = ANACHRONISM
    [0, 0, 0, 0, 1, 0],            # "telegram" = ANACHRONISM
    [0, 0, 0, 1, 0],               # "telegram" = ANACHRONISM
    [0, 0, 0, 0, 1, 0, 0],         # "quill" = ANACHRONISM
    [0, 0, 0, 1, 0],               # "telegram" = ANACHRONISM
    [0, 0, 0, 0, 1, 0],            # "telegraph" = ANACHRONISM
    [0, 0, 0, 0, 1, 0],            # "gramophone" = ANACHRONISM
    [0, 0, 1, 0, 0, 0],            # "frock" = ANACHRONISM
    [0, 1, 0, 0, 0],               # "telephony" = ANACHRONISM
    [0, 0, 0, 0, 0],               # All OK (contextual)
    [0, 0, 0, 0, 0, 0, 1],         # "quill" = ANACHRONISM
    [0, 0, 0, 0, 0, 0],            # "telegraph" = ANACHRONISM
    [0, 0, 0, 1, 0, 0, 0],         # "frock" = ANACHRONISM
    [0, 0, 0, 2, 0, 0],            # "dope" = STYLE_MISMATCH
    [0, 0, 0, 2, 0, 0],            # "lit" = STYLE_MISMATCH
    [0, 0, 0, 2, 0],               # "stan" = STYLE_MISMATCH
    [0, 0, 0, 0, 2, 0, 0],         # "rad" = STYLE_MISMATCH
    [0, 0, 0, 0, 0, 2],            # "dope" = STYLE_MISMATCH
    [0, 0, 0, 2, 0, 0],            # "lit" = STYLE_MISMATCH
    [0, 0, 0, 2, 0, 0],            # "cool" = STYLE_MISMATCH
    [0, 0, 0, 0],                  # All OK
    [0, 0, 0, 2, 0, 0],            # "jive" = STYLE_MISMATCH
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 0, 0],                  # All OK
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 0, 0],                  # All OK
    [0, 0, 0, 0, 0],               # All OK
    [0, 0, 0, 0, 0, 0],            # All OK
    [0, 0, 0, 0, 0, 0],            # All OK
    [0, 0, 0, 0, 0, 0],            # All OK (telegram OK in 1940)
    [0, 0, 0, 0, 0, 0],            # All OK (telegrams OK in 1918)
    [0, 0, 0, 0, 0, 0]             # All OK (telegraph OK in 20th war context)
]

def preprocess_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def suggest_alternative(word, timeframe):
    start_year = 1900 if timeframe == "20th" else 2000
    end_year = 1999 if timeframe == "20th" else 2099
    if word not in word2vec:
        return "unknown"
    if timeframe == "20th" and word == "lit":
        return "great"
    similar = word2vec.most_similar(word, topn=10)
    for syn, _ in similar:
        if syn in anachronism_dict and start_year <= anachronism_dict[syn]["year"] <= end_year:
            return syn
        elif syn not in anachronism_dict and syn not in ["telegrams", "telegrammed"]:
            if timeframe == "21st" and syn == "message":
                return syn
            if timeframe == "20th" and syn == "letter":
                return syn
            if syn != word:
                return syn
    synsets = wn.synsets(word)
    for syn in synsets[:3]:
        for lemma in syn.lemmas():
            syn_word = lemma.name()
            if syn_word in anachronism_dict and start_year <= anachronism_dict[syn_word]["year"] <= end_year:
                return syn_word
            elif syn_word not in anachronism_dict and syn_word not in ["telegrams", "telegrammed"]:
                return syn_word
    return "message" if timeframe == "21st" else "letter"

def predict_style(word, pos):
    return style_classifier.classify({"word": word, "pos": pos})

def analyze_file(file_path, timeframe, expected_style="neutral"):
    text = preprocess_text(file_path)
    doc = nlp(text)
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = bert_model(inputs)[0]
    predictions = tf.argmax(outputs, axis=-1).numpy()[0]
    words = tokenizer.tokenize(text)
    
    print(f"Predictions for {file_path}: {predictions.tolist()}")
    print(f"Tokens: {words}")
    
    issues = []
    seen_words = set()
    for i, (word, pred) in enumerate(zip(words, predictions[1:len(words)+1])):  # Skip [CLS]
        if pred == 1 and word not in seen_words:
            year = anachronism_dict.get(word, {}).get("year", 0)
            if timeframe == "20th" and word in ["telegram", "telegraph"] and year < 1980:
                continue
            issues.append((word, year, "anachronism (BERT)", suggest_alternative(word, timeframe)))
            seen_words.add(word)
        elif pred == 2 and word not in seen_words:
            year = anachronism_dict.get(word, {}).get("year", 0)
            predicted_style = predict_style(word, nlp(word).doc[0].pos_)
            issues.append((word, year, f"style mismatch (BERT, expected {expected_style}, got {predicted_style})", suggest_alternative(word, timeframe)))
            seen_words.add(word)

    for token in doc:
        word = token.text
        if word in anachronism_dict and word not in seen_words:
            word_info = anachronism_dict[word]
            if word_info["style"] != expected_style and word_info["style"] in ["slang", "casual"] and expected_style == "formal":
                issues.append((word, word_info["year"], f"style mismatch (rule, expected {expected_style}, got {word_info['style']})", suggest_alternative(word, timeframe)))
                seen_words.add(word)
            elif (timeframe == "20th" and word_info["year"] >= 2000) or \
                 (timeframe == "21st" and word_info["year"] < 1900):
                issues.append((word, word_info["year"], "anachronism (rule)", suggest_alternative(word, timeframe)))
                seen_words.add(word)

    valid_words = [token.text for token in doc if token.text in word2vec and token.text in anachronism_dict]
    if valid_words:
        avg_embedding = np.mean([word2vec[w] for w in valid_words], axis=0)
        for token in doc:
            word = token.text
            if word in word2vec and word in anachronism_dict and word not in seen_words:
                similarity = word2vec.cosine_similarities(word2vec[word], [avg_embedding])[0]
                if similarity < 0.7:
                    year = anachronism_dict.get(word, {}).get("year", 0)
                    issues.append((word, year, "consistency mismatch", suggest_alternative(word, timeframe)))

    results = pd.DataFrame(issues, columns=["word", "usage_year", "issue", "suggestion"])
    return results

def main():
    parser = argparse.ArgumentParser(description="AI Tool for Writers: Detect anachronisms and dialogue issues.")
    parser.add_argument("file", help="Path to the input .txt file")
    parser.add_argument("timeframe", choices=["20th", "21st"], help="Century the text is set in")
    parser.add_argument("--style", choices=["formal", "casual", "slang", "neutral"], default="neutral", help="Expected dialogue style")
    args = parser.parse_args()

    # Load and train BERT here
    global bert_model, style_classifier
    bert_model = TFBertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)
    train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="tf")
    max_length = train_encodings['input_ids'].shape[1]
    padded_labels = []
    for labels in train_labels:
        padded = [0] + labels + [0] * (max_length - len(labels) - 1)
        padded_labels.append(padded[:max_length])
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), padded_labels)).batch(1)  # Reduced batch size to 1
    bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    bert_model.fit(train_dataset, epochs=10)
    bert_model.save_pretrained("models/bert_trained")

    # Train Naive Bayes for style
    style_data = [("telegram", "NOUN", "formal"), ("phonograph", "NOUN", "neutral"), ("selfie", "NOUN", "casual"), ("lit", "ADJ", "slang"), ("message", "NOUN", "neutral"), ("cool", "ADJ", "slang"), ("dope", "ADJ", "slang")]
    style_features = [({"word": w, "pos": p}, s) for w, p, s in style_data]
    style_classifier = nltk.classify.NaiveBayesClassifier.train(style_features)

    os.makedirs("data/output", exist_ok=True)
    results = analyze_file(args.file, args.timeframe, args.style)
    if results.empty:
        print(f"No issues detected in {args.file} (set in {args.timeframe} century, style: {args.style}).")
    else:
        print(f"Issues detected in {args.file} (set in {args.timeframe} century, style: {args.style}):")
        print(results.to_string(index=False))
        results.to_csv("data/output/anachronism_report.csv", index=False)
        print("Results saved to data/output/anachronism_report.csv")

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    main()
