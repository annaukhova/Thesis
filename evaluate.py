import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("../models")

def encode_texts(texts, tokenizer):
    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="tf")

test_data = pd.read_csv("C:/Users/Hp/Desktop/Thesis/data/test_data.csv")
test_encodings = encode_texts(test_data["text"], tokenizer)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_data["label"].tolist()
))

# Compilation before the evaluation as otherwise it gives me an error
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

results = model.evaluate(test_dataset.batch(2))
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")


