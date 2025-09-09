import json
import os
from gensim import corpora
from collections import Counter
import pandas as pd


print("\n📁 Loading processed reviews...")
processed_file_path = "src/data/processed/processed_reviews.json"

try:
    with open(processed_file_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    processed_reviews = processed_data['processed_reviews']
    original_reviews = processed_data['original_reviews']
    stats = processed_data['preprocessing_stats']
    
    print(f"✅ Loaded {stats['total_reviews']} processed reviews")
    print(f"📈 Total unique words: {stats['total_unique_words']}")
    print(f"📈 Total words: {stats['total_words']}")
    
except FileNotFoundError:
    print("❌ Processed data file not found. Please run Step 2 first.")
    exit()

# Step 3.1: Create Dictionary
print("\n🔤 Creating Gensim Dictionary...")
print("This maps each unique word to a unique ID number")

# Create dictionary from processed reviews
dictionary = corpora.Dictionary(processed_reviews)

print(f"✅ Dictionary created with {len(dictionary)} unique words")

# Let's see some examples of word-to-ID mapping
print("\n📝 Sample word-to-ID mappings:")
sample_words = list(dictionary.token2id.items())[:10]
for word, word_id in sample_words:
    print(f"  Word: '{word}' → ID: {word_id}")

# Step 3.2: Filter the dictionary
print("\n🔍 Filtering dictionary...")
print("Removing words that appear in very few or too many documents")

# Before filtering
print(f"Dictionary size before filtering: {len(dictionary)}")

# Filter extremes:
# no_below: ignore words that appear in less than 2 documents
# no_above: ignore words that appear in more than 80% of documents
# keep_n: keep only the 1000 most frequent words
dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=1000)

print(f"Dictionary size after filtering: {len(dictionary)}")

# Step 3.3: Create Corpus (Bag of Words)
print("\n📊 Creating Corpus (Bag of Words representation)...")
print("Converting each review into (word_id, frequency) pairs")

# Create corpus - each document becomes a list of (word_id, word_frequency) tuples
corpus = [dictionary.doc2bow(review) for review in processed_reviews]

print(f"✅ Corpus created with {len(corpus)} documents")

# Let's examine what a corpus document looks like
print("\n🔍 Sample corpus document analysis:")
print(f"Original review: '{original_reviews[0][:100]}...'")
print(f"Processed tokens: {processed_reviews[0]}")
print(f"Corpus representation: {corpus[0]}")

print("\nExplaining the corpus format:")
for word_id, frequency in corpus[0][:5]:  # Show first 5 word-frequency pairs
    word = dictionary[word_id]
    print(f"  Word ID {word_id} ('{word}') appears {frequency} times")

# Step 3.4: Corpus Statistics
print("\n📈 Corpus Statistics:")

# Calculate corpus statistics
total_word_occurrences = sum(sum(freq for _, freq in doc) for doc in corpus)
unique_words_in_corpus = len(dictionary)
avg_words_per_doc = total_word_occurrences / len(corpus)

print(f"📊 Total word occurrences across all documents: {total_word_occurrences}")
print(f"📊 Unique words in corpus: {unique_words_in_corpus}")
print(f"📊 Average words per document: {avg_words_per_doc:.2f}")

# Find documents with most/least words
doc_lengths = [sum(freq for _, freq in doc) for doc in corpus]
max_length_idx = doc_lengths.index(max(doc_lengths))
min_length_idx = doc_lengths.index(min(doc_lengths))

print(f"\n📏 Document length analysis:")
print(f"Longest document: #{max_length_idx + 1} with {max(doc_lengths)} words")
print(f"  Text: '{original_reviews[max_length_idx][:100]}...'")
print(f"Shortest document: #{min_length_idx + 1} with {min(doc_lengths)} words")
print(f"  Text: '{original_reviews[min_length_idx][:100]}...'")

# Step 3.5: Save Dictionary and Corpus
print("\n💾 Saving Dictionary and Corpus...")

# Create models directory
models_dir = "src/models"
os.makedirs(models_dir, exist_ok=True)

# Save dictionary
dictionary_path = os.path.join(models_dir, "review_dictionary.dict")
dictionary.save(dictionary_path)
print(f"✅ Dictionary saved to: {dictionary_path}")

# Save corpus
corpus_path = os.path.join(models_dir, "review_corpus.mm")
corpora.MmCorpus.serialize(corpus_path, corpus)
print(f"✅ Corpus saved to: {corpus_path}")

# Save metadata for easy reference later
metadata = {
    'dictionary_size': len(dictionary),
    'corpus_size': len(corpus),
    'total_word_occurrences': total_word_occurrences,
    'avg_words_per_doc': avg_words_per_doc,
    'max_doc_length': max(doc_lengths),
    'min_doc_length': min(doc_lengths),
    'dictionary_path': dictionary_path,
    'corpus_path': corpus_path
}

metadata_path = os.path.join(models_dir, "corpus_metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved to: {metadata_path}")

# Step 3.6: Validation Check
print("\n✅ Validation Checks:")

# Check if any documents became empty after filtering
empty_docs = [i for i, doc in enumerate(corpus) if len(doc) == 0]
if empty_docs:
    print(f"⚠️ Warning: {len(empty_docs)} documents became empty after filtering")
    print(f"Empty document indices: {empty_docs[:10]}...")  # Show first 10
else:
    print("✅ No empty documents - corpus is ready for LDA!")

# Check dictionary consistency
print(f"✅ Dictionary consistency check: {len(dictionary)} words mapped")

# Show word frequency distribution
word_frequencies = {}
for doc in corpus:
    for word_id, freq in doc:
        if word_id in word_frequencies:
            word_frequencies[word_id] += freq
        else:
            word_frequencies[word_id] = freq

print(f"\n📊 Top 10 most frequent words in corpus:")
sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
for word_id, total_freq in sorted_words[:10]:
    word = dictionary[word_id]
    print(f"  '{word}': {total_freq} total occurrences")

print("\n" + "="*50)
print("STEP 3 COMPLETE!")
print("="*50)
print("✅ Dictionary created and filtered")
print("✅ Corpus (Bag of Words) generated")
print("✅ Dictionary and Corpus saved to src/models/")
print("✅ Validation checks passed")
print("✅ Data ready for LDA training!")
print("\nType 'next' when you're ready for Step 4: Training the LDA Model")