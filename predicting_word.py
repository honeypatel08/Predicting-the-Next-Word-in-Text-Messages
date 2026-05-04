# NEXT WORD PREDICTION MODEL
# Datasets: SMS Spam Collection + Multi-Turn Chatbot Conversation
# Models: Bigram (Baseline) + LSTM (Deep Learning)

from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# DATA LOADING - Load each dataset, extract only the text column, tag it with a source label, then them into one combined DataFrame.

def load_sms(filepath="spam.csv"):
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v2']].copy()
    df.columns = ['Message']
    df['source'] = 'sms'
    return df


def load_chatbot(filepath="chatbot_conversations.csv",  sample_size=20000 ):
    df = pd.read_csv(filepath, encoding='utf-8')
    df = df[['message']].copy()
    df.columns = ['Message']
    df['source'] = 'chatbot'
    # Sample a manageable subset so training finishes in minutes not hours
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Chatbot dataset sampled down to {sample_size} rows (from {len(df) + (len(df) - sample_size):,})")

    return df


# Load both datasets and combine into one DataFrame
df_sms = load_sms("spam.csv")
df_chat = load_chatbot("chatbot_conversations.csv")
df = pd.concat([df_sms, df_chat], ignore_index=True)

print(f"Loaded {len(df_sms)} SMS rows and {len(df_chat)} chatbot rows")
print(f"Combined total: {len(df)} rows\n")


# Data CLEANING
# Standardise every message so the model sees consistent tokens:
#   • lowercase everything, replace newlines with a space, strip punctuation / special characters (keep letters, digits, spaces), collapse multiple spaces into one

def clean_text(text):
    text = str(text).lower()                  
    text = re.sub(r'\n', ' ', text)            
    text = re.sub(r'[^a-z0-9\s]', '', text)    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['Message'] = df['Message'].apply(clean_text)

# Drop NA rows 
df = df.dropna(subset=['Message'])
df = df[df['Message'].str.strip() != ""]
df = df.reset_index(drop=True)

# Save cleaned combined dataset for later reference
df.to_csv("combined_clean.csv", index=False)

print("Sample of cleaned messages:")
print(df.head(5))
print()


# EXPLORATORY DATA ANALYSIS (EDA)
# We compare SMS vs chatbot on: message counts, average message length, top-10 most common words (overall + per source). message length distribution

# Message count per source
source_counts = df['source'].value_counts()
plt.figure()
source_counts.plot(kind='bar', color=['steelblue', 'coral'])
plt.title("Number of Messages per Dataset")
plt.xlabel("Dataset")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Average message length per source 
df['word_count'] = df['Message'].apply(lambda x: len(x.split()))
avg_len = df.groupby('source')['word_count'].mean()
print("Average message length (words) per source:")
print(avg_len, "\n")

# Combine Top 10 words — overall
all_words    = " ".join(df["Message"]).split()
common_words = Counter(all_words).most_common(10)
words_list   = [w[0] for w in common_words]
counts_list  = [w[1] for w in common_words]

plt.figure()
plt.bar(words_list, counts_list, color='steelblue')
plt.title("Top 10 Most Common Words (Combined)")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 words — per source (Spam or conversation)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'sms': 'steelblue', 'chatbot': 'coral'}

for ax, src in zip(axes, ['sms', 'chatbot']):
    subset_words = " ".join(df[df['source'] == src]["Message"]).split()
    top10 = Counter(subset_words).most_common(10)
    w, c = zip(*top10)
    ax.bar(w, c, color=colors[src])
    ax.set_title(f"Top 10 Words — {src.upper()}")
    ax.set_xlabel("Word")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(w, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Message length distribution
plt.figure()
plt.hist(df['word_count'], bins=30, color='steelblue', edgecolor='white')
plt.title("Message Length Distribution (Combined)")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# BIGRAM BASELINE MODEL - next word by looking at only the previous word.
# It counts how often word B follows word A across the entire corpus, then
# returns the most frequent follower.
# This acts as our simple baseline before training the LSTM.
corpus_text = " ".join(df["Message"])
corpus_words = corpus_text.split()

# Create all (word1, word2) consecutive pairs
bigrams = list(zip(corpus_words[:-1], corpus_words[1:]))

# Build the bigram frequency table: bigram_model["hello"]["world"] = number of times "world" follows "hello"
bigram_model = defaultdict(Counter)
for w1, w2 in bigrams:
    bigram_model[w1][w2] += 1

def predict_bigram(word):
    if word in bigram_model:
        return bigram_model[word].most_common(1)[0][0]
    return "no_prediction"


def generate_text_bigram(start_word, n=5):
    result  = [start_word]
    current = start_word
    for _ in range(n):
        next_w = predict_bigram(current)
        if next_w == "no_prediction":
            break
        result.append(next_w)
        current = next_w
    return " ".join(result)

# Quick demo 
print("Bigram prediction for 'i':", predict_bigram("i"))
print("Generated sentence from 'i':", generate_text_bigram("i"))
print()

print("Bigram Prediction — Interactive (5 words) - To Predict next word in phrase:")
for i in range(5):
    user_input = input(f"  Enter word {i+1}: ").lower().strip()
    if user_input == "":
        print("  (empty — skipped)")
        continue
    print("  Prediction:", predict_bigram(user_input))
print()

print("Bigram Prediction — Interactive (5 words) - To Predict phrase:")
for i in range(5):
    user_input = input(f"  Enter word {i+1}: ").lower().strip()
    if user_input == "":
        print("  (empty — skipped)")
        continue
    print("  Prediction:", generate_text_bigram(user_input))
print()


# TOKENISATION & SEQUENCE GENERATION
VOCAB_SIZE  = 8000  # increase to cover chatbot vocabulary,
WINDOW_SIZE = 20 

# Tokeniser: maps each word to a unique integer, keeping only the top VOCAB_SIZE words
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Message'])

sequences = []

for line in df['Message']:
    token_list = tokenizer.texts_to_sequences([line])[0]  

    
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[max(0, i - WINDOW_SIZE): i + 1]
        sequences.append(n_gram_seq)

MAX_SEQ_LEN     = WINDOW_SIZE
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='pre')
padded_sequences = np.array(padded_sequences)

X = padded_sequences[:, :-1] 
y = padded_sequences[:, -1]

print(f"Total training sequences : {len(sequences)}")
print(f"Sequence length (X)      : {X.shape[1]}")
print(f"Vocabulary size          : {VOCAB_SIZE}\n")


# TRAIN / TEST SPLIT Hold out 20% of sequences for evaluation so the model is tested on data it has never seen during training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}\n")

print("Note: Training may take several minutes due to dataset size. Be Patient 🙂\n")

# LSTM MODEL DEFINITION
lstm_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128),        
    LSTM(256, return_sequences=True),                       
    Dropout(0.3),                                        
    LSTM(128),                                       
    Dropout(0.3),                         
    Dense(128, activation='relu'),                             
    Dense(VOCAB_SIZE, activation='softmax')                  
])

# sparse_categorical_crossentropy expects integer labels (not one-hot encoded)
lstm_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

lstm_model.summary()


# MODEL TRAINING- EarlyStopping monitors validation loss and stops training if it stops improving for 3 consecutive epochs, then restores the best weights.
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = lstm_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

#  Plot training history 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title("Model Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title("Model Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.show()



# EVALUATION
# Metrics:Accuracy  — fraction of test samples where top-1 prediction is correct, Perplexity — e^(cross-entropy loss); lower = better; measures how
#                  "surprised" the model is by the test sequence and  Bigram accuracy — baseline comparison using simple bigram look-up

loss, accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"\nLSTM Test Accuracy : {accuracy:.4f}")


def compute_perplexity(model, X, y):
    ce_loss = model.evaluate(X, y, verbose=0)[0]   # cross-entropy
    return np.exp(ce_loss)


print(f"LSTM Perplexity    : {compute_perplexity(lstm_model, X_test, y_test):.2f}")


def bigram_accuracy(word_list):
    correct = 0
    total   = 0
    for i in range(len(word_list) - 1):
        if predict_bigram(word_list[i]) == word_list[i + 1]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0


sample_words = corpus_words[:5000]
print(f"Bigram Accuracy    : {bigram_accuracy(sample_words):.4f}\n")

# ── Comparison table ──────────────────────────────────────────────────────────
print("=" * 45)
print(f"{'Model':<15} {'Accuracy':>10} {'Perplexity':>15}")
print("-" * 45)
print(f"{'Bigram':<15} {bigram_accuracy(sample_words):>10.4f} {'N/A':>15}")
print(f"{'LSTM':<15} {accuracy:>10.4f} {compute_perplexity(lstm_model, X_test, y_test):>15.2f}")
print("=" * 45)


# PER-SOURCE EVALUATION
# Evaluate the LSTM separately on SMS and chatbot test samples.
# This shows which conversation type the model learned better.
# Rebuild source labels aligned with the sequences
# Each sequence came from a specific row; retrieve the source for each row index
source_per_seq = []
for idx, row in df.iterrows():
    line       = row['Message']
    token_list = tokenizer.texts_to_sequences([line])[0]
    n_seqs     = max(0, len(token_list) - 1)   # number of sequences from this row
    source_per_seq.extend([row['source']] * n_seqs)

source_arr = np.array(source_per_seq)

# Align with padded_sequences (same ordering)
assert len(source_arr) == len(X), "Source array length mismatch — check sequence generation"

# Split source labels the same way we split X / y
_, source_test = train_test_split(source_arr, test_size=0.2, random_state=42)

print("\nPer-source LSTM evaluation:")
for src in ['sms', 'chatbot']:
    mask     = source_test == src
    X_src    = X_test[mask]
    y_src    = y_test[mask]
    if len(X_src) == 0:
        print(f"  {src}: no test samples found")
        continue
    l, a = lstm_model.evaluate(X_src, y_src, verbose=0)
    pp   = np.exp(l)
    print(f"  {src.upper():>10} — Accuracy: {a:.4f}  |  Perplexity: {pp:.2f}")


# PREDICTION 
# Two LSTM prediction modes:
#   • predict_next_word — returns the single most likely next word (greedy)
#   • predict_top_k     — returns the top-k candidates with probabilities

# Reverse lookup: integer index → word string
index_word = {v: k for k, v in tokenizer.word_index.items()}


def predict_next_word(text, model, tokenizer, max_len):
    text     = re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()
    sequence = tokenizer.texts_to_sequences([text])

    if not sequence or len(sequence[0]) == 0:
        return "no input"

    padded    = pad_sequences([sequence[0]], maxlen=max_len, padding='pre')
    probs     = lstm_model.predict(padded, verbose=0)[0]
    best_idx  = np.argmax(probs)
    return index_word.get(best_idx, "<UNK>")


def predict_top_k(text, model, tokenizer, max_len, k=3):
    text     = re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()
    sequence = tokenizer.texts_to_sequences([text])

    if not sequence or len(sequence[0]) == 0:
        return [("no input", 0.0)]

    padded      = pad_sequences([sequence[0]], maxlen=max_len, padding='pre')
    probs       = lstm_model.predict(padded, verbose=0)[0]
    top_indices = np.argsort(probs)[-k:][::-1]   # indices of top-k probabilities

    return [(index_word.get(i, "<UNK>"), float(probs[i])) for i in top_indices]



# INTERACTIVE DEMO
# user type any phrase and see:
# the single best next-word prediction (LSTM), the top-3 candidates with probabilities, the bigram model's prediction for comparison

print("\n=== INTERACTIVE NEXT-WORD PREDICTION ===")
print("Type a phrase and press Enter to see predictions.")
print("Type 'done' to quit.\n")

while True:
    user_text = input("Enter text: ").strip()

    if user_text.lower() == "done":
        print("Exiting. Goodbye!")
        break

    best   = predict_next_word(user_text, lstm_model, tokenizer, MAX_SEQ_LEN)
    top3   = predict_top_k(user_text, lstm_model, tokenizer, MAX_SEQ_LEN, k=3)
    bigram = predict_bigram(user_text.split()[-1])   # bigram uses only the last word

    print(f"  LSTM best prediction : {best}")
    print(f"  LSTM top-3           : {[(w, f'{p:.3f}') for w, p in top3]}")
    print(f"  Bigram prediction    : {bigram}")
    print()