import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def encode_texts(texts, tokenizer):
    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="tf")

train_data = pd.read_csv("C:/Users/Hp/Desktop/Thesis/data/train_data.csv")
test_data = pd.read_csv("C:/Users/Hp/Desktop/Thesis/data/test_data.csv")

train_encodings = encode_texts(train_data["text"], tokenizer)
test_encodings = encode_texts(test_data["text"], tokenizer)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_data["label"].tolist()
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_data["label"].tolist()
))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# Using test_dataset as validation data due to small dataset size for now
model.fit(train_dataset.shuffle(1000).batch(2), epochs=3, validation_data=test_dataset.batch(2))
model.save_pretrained("../models")

