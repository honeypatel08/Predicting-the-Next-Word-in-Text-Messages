import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Phase 1: Data Preprocessing aand Cleaning
# Function to clean the text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v2']]
df.columns = ['Message']

df['Message'] = df['Message'].apply(clean_text)
df = df.dropna(subset=['Message'])
df = df[df['Message'].str.strip() != ""]
df = df.reset_index(drop=True)

df.to_csv("spam_clean.csv", index=False)

print(df.head(5))

# Phase 2: Tokenization and Seq Generartion 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Message'])
# text -> sequences
sequences = []

for line in df['Message']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # Create n-gram sequences
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        sequences.append(n_gram_seq)

# Pad sequences
max_seq_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

# Split into input (X) and output (y)
padded_sequences = np.array(padded_sequences)
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]
vocab_size = len(tokenizer.word_index) + 1

print("Total sequences:", len(sequences))
print("Max sequence length:", max_seq_len)
print("Vocab size:", vocab_size)