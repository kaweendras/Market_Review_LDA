import re
import pandas as pd
import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Fix for newer NLTK versions - download the additional required data
print("🔧 Downloading additional NLTK data for newer versions...")
try:
    nltk.download('punkt_tab', quiet=True)  # This is what was missing!
    print("✅ Additional NLTK data downloaded successfully!")
except:
    print("⚠️ If you still get errors, try running: nltk.download('punkt_tab')")

# Initialize the tools we need
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

print("🧹 Setting up text preprocessing tools...")
print(f"✅ Loaded {len(stop_words)} English stop words")
print("✅ Initialized lemmatizer")

# Let's see what stop words look like
print(f"\nSample stop words: {list(stop_words)[:20]}...")

# Main preprocessing function for all reviews
def preprocess_text(text):
    """Clean and preprocess review text for LDA analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Process all our sample reviews
print("\n" + "="*60)
print("PROCESSING ALL SAMPLE REVIEWS")
print("="*60)

# Load reviews from CSV file
print("📁 Loading reviews from CSV file...")
csv_path = "src/data/raw/sample_reviews.csv"
df = pd.read_csv(csv_path)
sample_reviews = df['review_text'].tolist()
print(f"✅ Loaded {len(sample_reviews)} reviews from {csv_path}")

processed_reviews = []
for i, review in enumerate(sample_reviews):
    processed = preprocess_text(review)
    processed_reviews.append(processed)
    print(f"Review {i+1}: {processed}")

print(f"\n✅ Processed {len(processed_reviews)} reviews!")

# Save processed data to the processed directory
print("\n💾 Saving processed data...")
processed_dir = "src/data/processed"
os.makedirs(processed_dir, exist_ok=True)

# Save processed reviews as JSON for easy loading later
processed_data = {
    'processed_reviews': processed_reviews,
    'original_reviews': sample_reviews,
    'preprocessing_stats': {
        'total_reviews': len(processed_reviews),
        'total_unique_words': len(set([word for review in processed_reviews for word in review])),
        'total_words': sum(len(review) for review in processed_reviews)
    }
}

processed_file_path = os.path.join(processed_dir, "processed_reviews.json")
with open(processed_file_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

print(f"✅ Processed data saved to: {processed_file_path}")

# Also save as CSV for easy viewing
processed_df = pd.DataFrame({
    'review_id': range(1, len(sample_reviews) + 1),
    'original_text': sample_reviews,
    'processed_tokens': [' '.join(tokens) for tokens in processed_reviews],
    'token_count': [len(tokens) for tokens in processed_reviews]
})

csv_file_path = os.path.join(processed_dir, "processed_reviews.csv")
processed_df.to_csv(csv_file_path, index=False, encoding='utf-8')
print(f"✅ Processed data also saved as CSV: {csv_file_path}")

# Let's analyze what words appear most frequently
from collections import Counter

# Flatten all words from all reviews
all_words = [word for review in processed_reviews for word in review]
word_counts = Counter(all_words)

print("\n📊 Most common words after preprocessing:")
for word, count in word_counts.most_common(10):
    print(f"  '{word}': {count} times")

print(f"\n📈 Total unique words: {len(word_counts)}")
print(f"📈 Total words: {len(all_words)}")

# Let's also check if we have any empty reviews (which would be a problem)
empty_reviews = [i for i, review in enumerate(processed_reviews) if len(review) == 0]
if empty_reviews:
    print(f"⚠️ Warning: {len(empty_reviews)} empty reviews after preprocessing")
else:
    print("✅ No empty reviews - good!")

print("\n" + "="*50)
print("STEP 2 COMPLETE!")
print("="*50)
print("✅ Text preprocessing function created")
print("✅ Sample reviews processed and cleaned")
print("✅ Word frequency analysis completed")
print("✅ Processed data saved to src/data/processed/")
print("✅ Data ready for LDA modeling")
print("\nType 'next' when you're ready for Step 3: Creating Dictionary and Corpus")