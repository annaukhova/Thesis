import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

print("Current Working Directory:", os.getcwd())

# Dictionary of anachronistic words
words_20 = ["selfie", "cringe", "lit"]
words_21 = ["telegram", "phonograph", "telegraph"]

def preprocess_text(text, keywords):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    flagged_words = [word for word in words if word in keywords]
    return ' '.join(words), flagged_words

def analyze_text(file_path, label, keywords):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    text, flagged_words = preprocess_text(text, keywords)
    return text, label, flagged_words

file1_path = "C:/Users/Hp/Desktop/Thesis/data/file1.txt"
file2_path = "C:/Users/Hp/Desktop/Thesis/data/file2.txt"
text1, label1, flagged_words_20th = analyze_text(file1_path, 0, words_20)
text2, label2, flagged_words_21st = analyze_text(file2_path, 1, words_21)

data = pd.DataFrame({
    "text": [text1, text2],
    "label": [label1, label2],
    "flagged_words": [repr(flagged_words_20th), repr(flagged_words_21st)]
})

# Way to avoid empty training or test sets
train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)

# Due to the small dataset, skip validation split for now
train_data.to_csv("C:/Users/Hp/Desktop/Thesis/data/train_data.csv", index=False)
test_data.to_csv("C:/Users/Hp/Desktop/Thesis/data/test_data.csv", index=False)

