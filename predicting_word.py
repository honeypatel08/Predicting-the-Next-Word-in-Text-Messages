from collections import Counter

import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


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

# BIGRAM BASELINE MODEL (N-gram model)
text = " ".join(df["Message"])
words = text.split()

bigrams = list(zip(words[:-1], words[1:]))

model_bigram = defaultdict(Counter)

for w1, w2 in bigrams:
    model_bigram[w1][w2] += 1

def predict_bigram(word):
    if word in model_bigram:
        return model_bigram[word].most_common(1)[0][0]
    else:
        return "no_prediction"

print("Bigram test:", predict_bigram("i"))

# Test the bigram model
def generate_text(start_word, n=5):
    result = [start_word]
    current = start_word
    
    for _ in range(n):
        next_word = predict_bigram(current)
        if next_word == "no_prediction":
            break
        result.append(next_word)
        current = next_word
    
    return " ".join(result)

print("\nBigram Prediction Demo (Enter a word (one word only) to start): ")
# To change number of input times, modify the range in the for loop below. 

for i in range(5):
    user_input = input(f"\nEnter word {i+1}: ").lower().strip()
    
    if user_input == "":
        print("Empty input, try again")
        continue
    
    output = generate_text(user_input)
    print("Prediction:", output)


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
# max_seq_len = max(len(seq) for seq in sequences)
max_seq_len = 20

padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

# Split into input (X) and output (y)
padded_sequences = np.array(padded_sequences)
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]
vocab_size = len(tokenizer.word_index) + 1

# Basic stats about the data and EDA Section 
print("Total sequences:", len(sequences))
print("Max sequence length:", max_seq_len)
print("Vocab size:", vocab_size)


all_words = " ".join(df["Message"]).split()
common_words = Counter(all_words).most_common(10)

words = [w[0] for w in common_words]
counts = [w[1] for w in common_words]

plt.figure()
plt.bar(words, counts)
plt.title("Top 10 Most Common Words")
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

sentence_lengths = df["Message"].apply(lambda x: len(x.split()))

plt.figure()
plt.hist(sentence_lengths, bins=20)
plt.title("Sentence Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()