import pandas as pd
import re

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