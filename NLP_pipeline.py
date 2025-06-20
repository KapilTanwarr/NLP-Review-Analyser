import pandas as pd
import nltk
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic

# 📥 Download required lexicons
nltk.download("vader_lexicon")

# 📂 Load data and rename 'Text' column to 'review'
DATA_PATH = "C:\Users\Dell\Downloads\Reviews.csv"
df = pd.read_csv(DATA_PATH)

# Ensure 'Text' column exists
if "Text" not in df.columns:
    raise KeyError("The 'Text' column was not found in your CSV file.")

# Rename to match pipeline format
df = df.rename(columns={"Text": "review"})

# Drop missing reviews
df = df.dropna(subset=["review"]).reset_index(drop=True)

# (Optional) Use sample for faster testing
# df = df.sample(500).reset_index(drop=True)

# 🧹 Clean text
def clean(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

df["cleaned"] = df["review"].apply(clean)

# 🔍 Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def label(compound):
    if compound > 0.05:
        return "Positive"
    elif compound < -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_score"] = df["cleaned"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment"] = df["sentiment_score"].apply(label)

# 🧠 Topic Modeling
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(df["cleaned"])
df["topic"] = topics

# 💾 Save enriched data
df.to_csv(DATA_PATH, index=False)
print("✔️  Sentiment + topic columns written back to data/reviews.csv")




