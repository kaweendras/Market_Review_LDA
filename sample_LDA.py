import re
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

sample_reviews = [
        "The battery life is terrible, dies within 2 hours of use. Very disappointed.",
        "Amazing camera quality! Takes stunning photos even in low light. Love it!",
        "Customer service was unhelpful when I had issues. Took forever to get response.",
        "Great value for money. Battery lasts all day and camera is decent.",
        "Shipping was super fast, arrived next day. Product quality is excellent.",
        "The camera features are outstanding. Video recording is crystal clear.",
        "Battery performance is poor. Needs constant charging throughout the day.",
        "Customer support team was very helpful and resolved my issue quickly.",
        "Expensive but worth it. The camera and battery life exceeded expectations.",
        "Delivery was delayed by a week. Product is okay but packaging was damaged.",
        "The camera app is intuitive and easy to use. Battery life is acceptable.",
        "Poor customer service experience. Staff seemed uninterested in helping.",
        "Fast shipping and great packaging. The camera quality is phenomenal.",
        "Battery drains too quickly during video calls. Otherwise decent phone.",
        "Customer service went above and beyond to help with my problem.",
        "Overpriced for what you get. Battery life and camera are just average.",
        "The camera's night mode is incredible. Battery easily lasts a full day.",
        "Shipping took too long but the product quality makes up for it.",
        "Customer service response time needs improvement. Phone is good though.",
        "Best camera I've used on a phone. Battery life is surprisingly good too."
    ]

processed_reviews = []
for i, review in enumerate(sample_reviews):
    processed = preprocess_text(review)
    processed_reviews.append(processed)
    print(f"Review {i+1}: {processed}")

print(f"\n✅ Processed {len(processed_reviews)} reviews!")

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
print("✅ Data ready for LDA modeling")
print("\nType 'next' when you're ready for Step 3: Creating Dictionary and Corpus")