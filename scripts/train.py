import json
import os
from gensim import corpora
from gensim.models import LdaModel
import pandas as pd

print("\n" + "="*60)
print("STEP 4: TRAINING THE LDA MODEL")
print("="*60)

# Step 4.1: Load Dictionary and Corpus
print("\n📁 Loading Dictionary and Corpus...")

models_dir = "src/models"
dictionary_path = os.path.join(models_dir, "review_dictionary.dict")
corpus_path = os.path.join(models_dir, "review_corpus.mm")
metadata_path = os.path.join(models_dir, "corpus_metadata.json")

try:
    # Load dictionary
    dictionary = corpora.Dictionary.load(dictionary_path)
    print(f"✅ Dictionary loaded: {len(dictionary)} words")
    
    # Load corpus
    corpus = corpora.MmCorpus(corpus_path)
    print(f"✅ Corpus loaded: {len(corpus)} documents")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✅ Metadata loaded: {metadata['avg_words_per_doc']:.1f} avg words per doc")
    
except FileNotFoundError as e:
    print(f"❌ Required files not found: {e}")
    print("Please run Step 3 first to create dictionary and corpus.")
    exit()

# Step 4.2: Set LDA Parameters
print("\n⚙️ Setting LDA Parameters...")

# Key parameters explained:
num_topics = 5          # Number of topics to discover
passes = 10            # Number of training iterations
alpha = 'auto'         # Document-topic density (auto = let algorithm decide)
eta = 'auto'           # Topic-word density (auto = let algorithm decide)
random_state = 42      # For reproducible results

print(f"📊 Number of topics: {num_topics}")
print(f"📊 Training passes: {passes}")
print(f"📊 Alpha (doc-topic density): {alpha}")
print(f"📊 Eta (topic-word density): {eta}")
print(f"📊 Random state: {random_state}")

# Step 4.3: Train the LDA Model
print(f"\n🚀 Training LDA model...")
print("This may take a moment...")

# Train the LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=random_state,
    passes=passes,
    alpha=alpha,
    eta=eta,
    per_word_topics=True  # This helps with topic analysis later
)

print("✅ LDA model training completed!")

# Step 4.4: Display Discovered Topics
print("\n🔍 DISCOVERED TOPICS:")
print("="*50)

topics = lda_model.print_topics(num_words=8)
topic_summaries = []

for topic_id, topic_string in topics:
    print(f"\n📋 Topic {topic_id + 1}:")
    print("-" * 20)
    
    # Parse the topic string to extract words and probabilities
    topic_words = []
    parts = topic_string.split(' + ')
    
    for part in parts:
        prob_str, word = part.split('*')
        probability = float(prob_str.strip())
        word = word.strip().replace('"', '')
        topic_words.append((word, probability))
        print(f"  {word}: {probability:.3f}")
    
    # Create a summary for this topic
    top_words = [word for word, _ in topic_words[:4]]
    topic_summaries.append({
        'topic_id': topic_id,
        'top_words': top_words,
        'all_words': topic_words
    })

# Step 4.5: Analyze Model Performance
print(f"\n📈 MODEL PERFORMANCE METRICS:")
print("="*40)

# Calculate perplexity (lower is better)
perplexity = lda_model.log_perplexity(corpus)
print(f"📊 Log Perplexity: {perplexity:.3f}")
print("   (Lower perplexity = better model fit)")

# Calculate coherence (we'll do a simple version)
print(f"📊 Model trained on {len(corpus)} documents")
print(f"📊 Vocabulary size: {len(dictionary)} words")

# Step 4.6: Analyze Topic Distribution in Documents
print(f"\n📄 TOPIC DISTRIBUTION ANALYSIS:")
print("="*40)

# Load original reviews for analysis
processed_file_path = "src/data/processed/processed_reviews.json"
with open(processed_file_path, 'r', encoding='utf-8') as f:
    processed_data = json.load(f)
original_reviews = processed_data['original_reviews']

# Analyze first few documents
print("\nSample document topic assignments:")
document_topics = []

for i in range(min(5, len(corpus))):  # Analyze first 5 documents
    doc_topics = lda_model.get_document_topics(corpus[i])
    
    # Find dominant topic
    if doc_topics:
        dominant_topic_id, dominant_prob = max(doc_topics, key=lambda x: x[1])
        
        print(f"\n📝 Review {i+1}:")
        print(f"   Text: '{original_reviews[i][:80]}...'")
        print(f"   Dominant Topic: {dominant_topic_id + 1} (probability: {dominant_prob:.3f})")
        print(f"   All topics: {[(tid+1, prob) for tid, prob in doc_topics]}")
        
        document_topics.append({
            'review_id': i,
            'dominant_topic': dominant_topic_id + 1,
            'dominant_probability': dominant_prob,
            'all_topics': doc_topics
        })

# Step 4.7: Topic Distribution Across All Documents
print(f"\n📊 OVERALL TOPIC DISTRIBUTION:")
print("="*40)

topic_counts = {i: 0 for i in range(num_topics)}
total_docs = 0

for doc_bow in corpus:
    doc_topics = lda_model.get_document_topics(doc_bow)
    if doc_topics:
        dominant_topic_id, _ = max(doc_topics, key=lambda x: x[1])
        topic_counts[dominant_topic_id] += 1
        total_docs += 1

print("Topic distribution across all reviews:")
for topic_id in range(num_topics):
    count = topic_counts[topic_id]
    percentage = (count / total_docs) * 100 if total_docs > 0 else 0
    top_words = ', '.join(topic_summaries[topic_id]['top_words'][:3])
    print(f"  Topic {topic_id + 1} ({top_words}): {count} reviews ({percentage:.1f}%)")

# Step 4.8: Save the Trained Model
print(f"\n💾 SAVING TRAINED MODEL:")
print("="*30)

# Save the LDA model
model_path = os.path.join(models_dir, "lda_model")
lda_model.save(model_path)
print(f"✅ LDA model saved to: {model_path}")

# Convert numpy types to regular Python types for JSON serialization
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python types"""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

# Save topic summaries and results
results = {
    'model_parameters': {
        'num_topics': num_topics,
        'passes': passes,
        'alpha': alpha,
        'eta': eta,
        'random_state': random_state
    },
    'model_performance': {
        'log_perplexity': float(perplexity),
        'vocabulary_size': len(dictionary),
        'corpus_size': len(corpus)
    },
    'topics': convert_numpy_types(topic_summaries),
    'topic_distribution': topic_counts,
    'sample_document_analysis': convert_numpy_types(document_topics)
}

results_path = os.path.join(models_dir, "lda_results.json")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✅ Analysis results saved to: {results_path}")

# Step 4.9: Quick Business Insights
print(f"\n💡 QUICK BUSINESS INSIGHTS:")
print("="*35)

print("Based on the discovered topics, you can now:")
print("  🎯 Identify main customer concerns and interests")
print("  📊 Quantify how often each topic is discussed")
print("  📝 Classify new reviews automatically")
print("  📈 Track topic trends over time")
print("  💌 Create targeted marketing campaigns")

# Suggest topic interpretations
print(f"\n🤔 Suggested topic interpretations:")
for i, summary in enumerate(topic_summaries):
    top_words = ', '.join(summary['top_words'])
    print(f"  Topic {i + 1}: {top_words}")
    
    # Simple heuristic interpretations
    words = summary['top_words']
    if any(word in ['battery', 'charge', 'power'] for word in words):
        print(f"    → Likely about: Battery Performance")
    elif any(word in ['camera', 'photo', 'picture'] for word in words):
        print(f"    → Likely about: Camera Quality")
    elif any(word in ['service', 'support', 'help'] for word in words):
        print(f"    → Likely about: Customer Service")
    elif any(word in ['price', 'money', 'cost', 'expensive'] for word in words):
        print(f"    → Likely about: Pricing/Value")
    elif any(word in ['delivery', 'shipping', 'fast'] for word in words):
        print(f"    → Likely about: Shipping/Delivery")
    else:
        print(f"    → Review the words to determine the theme")

print("\n" + "="*50)
print("STEP 4 COMPLETE!")
print("="*50)
print("✅ LDA model successfully trained")
print("✅ Topics discovered and analyzed")
print("✅ Model performance metrics calculated")
print("✅ Topic distribution analyzed")
print("✅ Model and results saved")
print("✅ Business insights generated")