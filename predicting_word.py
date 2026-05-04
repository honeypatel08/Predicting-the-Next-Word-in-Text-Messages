from collections import Counter

import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# new
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

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

# ADDED
print("Bigram test:", predict_bigram("i"))

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

#tokenizer = Tokenizer()
# change to
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Message'])
# text -> sequences

#add
window_size = 20

sequences = []

for line in df['Message']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # Create n-gram sequences
    
    for i in range(1, len(token_list)):
        #n_gram_seq = token_list[:i+1]
        # changed to
        n_gram_seq = token_list[max(0, i-window_size):i+1]
        sequences.append(n_gram_seq)

# Pad sequences
# max_seq_len = max(len(seq) for seq in sequences)
max_seq_len = 20

padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

# Split into input (X) and output (y)
padded_sequences = np.array(padded_sequences)
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]
#vocab_size = len(tokenizer.word_index) + 1
# change to
vocab_size = 5000

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

# ADDED

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LSTM Model

lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(256, return_sequences=True),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

lstm_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

lstm_model.summary()

# Train LSTM model

early_stop = EarlyStopping(monitor= 'val_loss', patience=3, restore_best_weights=True)

lstm_model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test), callbacks = [early_stop])

# Evaluation 

loss, acc = lstm_model.evaluate(X_test, y_test)
print("Test Accuracy:", acc) 

def perplexity(model, X, y):
    loss = model.evaluate(X, y, verbose=0)[0]
    return np.exp(loss)

print ("Perplexity:", perplexity(lstm_model, X_test, y_test))

# Index lookup 
index_word =  {v: k for k, v in tokenizer.word_index.items()}

# LSTM Prediction
def predict_next_word_lstm(text, model, tokenizer, max_len):
    # clean input
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text).strip()

    # convert to sequence of tokens
    sequence = tokenizer.texts_to_sequences([text])

    if not sequence or len(sequence[0]) == 0:
        return "no input"

    sequence = sequence[0]

    padded = pad_sequences([sequence], maxlen=max_len, padding='pre')

    probs = model.predict(padded, verbose=0)

    prob_index = np.argmax(probs)

    return index_word.get(prob_index, "no prediction")

# TOP - K prediction
def predict_top_k(text, model, tokenizer, max_len, k=3):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
   
    sq = tokenizer.texts_to_sequences([text])
    if not sq or len(sq[0]) == 0:
        return ["no input"]
    sq = sq[0]

    padded = pad_sequences([sq], maxlen=max_len, padding='pre')
    probability = model.predict(padded, verbose=0)[0]

    top_k_indices = np.argsort(probability)[-k:][::-1]

    return [index_word.get(i, "<UNK>") for i in top_k_indices]

   
# Bigram Accuracy
def bigram_accuracy(test_words):
    correct = 0
    total = 0

    for i in range(len(test_words) - 1):
        prediction = predict_bigram(test_words[i])
        if prediction ==test_words[i]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

print ("Bigram accuracy sample: ", bigram_accuracy(words[:5000]))

while True:
    user_text = input("\nEnter text (or type 'done'): ").strip()

    if user_text.lower() == "done":
        break

    print("Next word:", predict_next_word_lstm(user_text, lstm_model, tokenizer, max_seq_len))
    print("Top-3:", predict_top_k(user_text, lstm_model, tokenizer, max_seq_len))

