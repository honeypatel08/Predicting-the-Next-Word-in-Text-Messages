import pandas as pd
import re

# Function to extract the body of the email, removing headers
def extract_body(text):
    parts = text.split("\n\n", 1) 
    if len(parts) > 1:
        return parts[1]
    return text

# Remove new lines, punctuation, numbers, and extra spaces
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)          
    text = re.sub(r'[^a-z\s]', '', text)    
    text = re.sub(r'\s+', ' ', text)       
    return text.strip()

df = pd.read_csv("enron.csv")
df = df[['Message']]
df['Message'] = df['Message'].apply(extract_body)
df['Message'] = df['Message'].apply(clean_text)
df = df.dropna(subset=['Message'])
df = df[df['Message'].str.strip() != ""]
df = df.reset_index(drop=True)
df.to_csv("enron_clean.csv", index=False)

print(df.head(5))