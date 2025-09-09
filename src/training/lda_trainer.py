import json
import os
from gensim import corpora
from gensim.models import LdaModel
import pandas as pd


def load_dictionary_and_corpus(models_dir="src/models", verbose=True):
    """
    Load dictionary, corpus, and metadata from saved files.
    
    Args:
        models_dir (str): Directory containing the saved models
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (dictionary, corpus, metadata)
    """
    if verbose:
        print("\n📁 Loading Dictionary and Corpus...")
    
    dictionary_path = os.path.join(models_dir, "review_dictionary.dict")
    corpus_path = os.path.join(models_dir, "review_corpus.mm")
    metadata_path = os.path.join(models_dir, "corpus_metadata.json")
    
    try:
        # Load dictionary
        dictionary = corpora.Dictionary.load(dictionary_path)
        if verbose:
            print(f"✅ Dictionary loaded: {len(dictionary)} words")
        
        # Load corpus
        corpus = corpora.MmCorpus(corpus_path)
        if verbose:
            print(f"✅ Corpus loaded: {len(corpus)} documents")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if verbose:
            print(f"✅ Metadata loaded: {metadata['avg_words_per_doc']:.1f} avg words per doc")
        
        return dictionary, corpus, metadata
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required files not found: {e}. Please run corpus creation first.")


def create_lda_model(corpus, dictionary, num_topics=5, passes=10, alpha='auto', 
                     eta='auto', random_state=42, verbose=True):
    """
    Create and train an LDA model.
    
    Args:
        corpus: Gensim corpus data
        dictionary: Gensim dictionary
        num_topics (int): Number of topics to discover
        passes (int): Number of training iterations
        alpha: Document-topic density parameter
        eta: Topic-word density parameter
        random_state (int): Random seed for reproducibility
        verbose (bool): Whether to print detailed information
        
    Returns:
        gensim.models.LdaModel: Trained LDA model
    """
    if verbose:
        print("\n⚙️ Setting LDA Parameters...")
        print(f"📊 Number of topics: {num_topics}")
        print(f"📊 Training passes: {passes}")
        print(f"📊 Alpha (doc-topic density): {alpha}")
        print(f"📊 Eta (topic-word density): {eta}")
        print(f"📊 Random state: {random_state}")
        
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
    
    if verbose:
        print("✅ LDA model training completed!")
    
    return lda_model


def extract_topics(lda_model, num_words=8, verbose=True):
    """
    Extract and format topics from the trained LDA model.
    
    Args:
        lda_model: Trained LDA model
        num_words (int): Number of words to show per topic
        verbose (bool): Whether to print detailed information
        
    Returns:
        list: List of topic summaries with words and probabilities
    """
    if verbose:
        print("\n🔍 DISCOVERED TOPICS:")
        print("="*50)
    
    topics = lda_model.print_topics(num_words=num_words)
    topic_summaries = []
    
    for topic_id, topic_string in topics:
        if verbose:
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
            if verbose:
                print(f"  {word}: {probability:.3f}")
        
        # Create a summary for this topic
        top_words = [word for word, _ in topic_words[:4]]
        topic_summaries.append({
            'topic_id': topic_id,
            'top_words': top_words,
            'all_words': topic_words
        })
    
    return topic_summaries


def calculate_model_performance(lda_model, corpus, dictionary, verbose=True):
    """
    Calculate and display model performance metrics.
    
    Args:
        lda_model: Trained LDA model
        corpus: Corpus data
        dictionary: Dictionary object
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Performance metrics
    """
    if verbose:
        print(f"\n📈 MODEL PERFORMANCE METRICS:")
        print("="*40)
    
    # Calculate perplexity (lower is better)
    perplexity = lda_model.log_perplexity(corpus)
    
    performance_metrics = {
        'log_perplexity': float(perplexity),
        'vocabulary_size': len(dictionary),
        'corpus_size': len(corpus)
    }
    
    if verbose:
        print(f"📊 Log Perplexity: {perplexity:.3f}")
        print("   (Lower perplexity = better model fit)")
        print(f"📊 Model trained on {len(corpus)} documents")
        print(f"📊 Vocabulary size: {len(dictionary)} words")
    
    return performance_metrics


def analyze_document_topics(lda_model, corpus, original_reviews, num_samples=5, verbose=True):
    """
    Analyze topic distribution in sample documents.
    
    Args:
        lda_model: Trained LDA model
        corpus: Corpus data
        original_reviews (list): Original review texts
        num_samples (int): Number of sample documents to analyze
        verbose (bool): Whether to print detailed information
        
    Returns:
        list: List of document topic analyses
    """
    if verbose:
        print(f"\n📄 TOPIC DISTRIBUTION ANALYSIS:")
        print("="*40)
        print("\nSample document topic assignments:")
    
    document_topics = []
    
    for i in range(min(num_samples, len(corpus))):
        doc_topics = lda_model.get_document_topics(corpus[i])
        
        # Find dominant topic
        if doc_topics:
            dominant_topic_id, dominant_prob = max(doc_topics, key=lambda x: x[1])
            
            if verbose:
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
    
    return document_topics


def calculate_topic_distribution(lda_model, corpus, topic_summaries, verbose=True):
    """
    Calculate overall topic distribution across all documents.
    
    Args:
        lda_model: Trained LDA model
        corpus: Corpus data
        topic_summaries (list): Topic summaries from extract_topics
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Topic distribution counts
    """
    if verbose:
        print(f"\n📊 OVERALL TOPIC DISTRIBUTION:")
        print("="*40)
    
    num_topics = len(topic_summaries)
    topic_counts = {i: 0 for i in range(num_topics)}
    total_docs = 0
    
    for doc_bow in corpus:
        doc_topics = lda_model.get_document_topics(doc_bow)
        if doc_topics:
            dominant_topic_id, _ = max(doc_topics, key=lambda x: x[1])
            topic_counts[dominant_topic_id] += 1
            total_docs += 1
    
    if verbose:
        print("Topic distribution across all reviews:")
        for topic_id in range(num_topics):
            count = topic_counts[topic_id]
            percentage = (count / total_docs) * 100 if total_docs > 0 else 0
            top_words = ', '.join(topic_summaries[topic_id]['top_words'][:3])
            print(f"  Topic {topic_id + 1} ({top_words}): {count} reviews ({percentage:.1f}%)")
    
    return topic_counts


def save_trained_model(lda_model, results_data, models_dir="src/models", verbose=True):
    """
    Save the trained LDA model and analysis results.
    
    Args:
        lda_model: Trained LDA model
        results_data (dict): Analysis results to save
        models_dir (str): Directory to save models
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: File paths where data was saved
    """
    if verbose:
        print(f"\n💾 SAVING TRAINED MODEL:")
        print("="*30)
    
    # Save the LDA model
    model_path = os.path.join(models_dir, "lda_model")
    lda_model.save(model_path)
    if verbose:
        print(f"✅ LDA model saved to: {model_path}")
    
    # Save analysis results
    results_path = os.path.join(models_dir, "lda_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"✅ Analysis results saved to: {results_path}")
    
    return {
        'model_path': model_path,
        'results_path': results_path
    }


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def generate_business_insights(topic_summaries, verbose=True):
    """
    Generate business insights based on discovered topics.
    
    Args:
        topic_summaries (list): Topic summaries from extract_topics
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Business insights and topic interpretations
    """
    if verbose:
        print(f"\n💡 QUICK BUSINESS INSIGHTS:")
        print("="*35)
        
        print("Based on the discovered topics, you can now:")
        print("  🎯 Identify main customer concerns and interests")
        print("  📊 Quantify how often each topic is discussed")
        print("  📝 Classify new reviews automatically")
        print("  📈 Track topic trends over time")
        print("  💌 Create targeted marketing campaigns")
    
    # Generate topic interpretations
    interpretations = {}
    
    if verbose:
        print(f"\n🤔 Suggested topic interpretations:")
    
    for i, summary in enumerate(topic_summaries):
        top_words = ', '.join(summary['top_words'])
        words = summary['top_words']
        
        if verbose:
            print(f"  Topic {i + 1}: {top_words}")
        
        # Simple heuristic interpretations
        interpretation = "Review the words to determine the theme"
        
        if any(word in ['battery', 'charge', 'power'] for word in words):
            interpretation = "Battery Performance"
        elif any(word in ['camera', 'photo', 'picture'] for word in words):
            interpretation = "Camera Quality"
        elif any(word in ['service', 'support', 'help'] for word in words):
            interpretation = "Customer Service"
        elif any(word in ['price', 'money', 'cost', 'expensive'] for word in words):
            interpretation = "Pricing/Value"
        elif any(word in ['delivery', 'shipping', 'fast'] for word in words):
            interpretation = "Shipping/Delivery"
        
        interpretations[f"topic_{i+1}"] = {
            'top_words': top_words,
            'interpretation': interpretation
        }
        
        if verbose:
            print(f"    → Likely about: {interpretation}")
    
    return interpretations


def train_lda_pipeline(models_dir="src/models",
                      processed_file_path="src/data/processed/processed_reviews.json",
                      num_topics=5,
                      passes=10,
                      alpha='auto',
                      eta='auto',
                      random_state=42,
                      num_words=8,
                      num_samples=5,
                      verbose=True):
    """
    Complete pipeline for training LDA model and analyzing results.
    
    Args:
        models_dir (str): Directory containing dictionary and corpus files
        processed_file_path (str): Path to processed reviews JSON file
        num_topics (int): Number of topics to discover
        passes (int): Number of training iterations
        alpha: Document-topic density parameter
        eta: Topic-word density parameter
        random_state (int): Random seed for reproducibility
        num_words (int): Number of words to show per topic
        num_samples (int): Number of sample documents to analyze
        verbose (bool): Whether to print detailed progress
        
    Returns:
        dict: Complete training results and analysis
    """
    if verbose:
        print("\n" + "="*60)
        print("STARTING LDA MODEL TRAINING PIPELINE")
        print("="*60)
    
    # Step 1: Load dictionary and corpus
    dictionary, corpus, metadata = load_dictionary_and_corpus(models_dir, verbose)
    
    # Step 2: Train LDA model
    lda_model = create_lda_model(
        corpus, dictionary, num_topics, passes, alpha, eta, random_state, verbose
    )
    
    # Step 3: Extract topics
    topic_summaries = extract_topics(lda_model, num_words, verbose)
    
    # Step 4: Calculate performance metrics
    performance_metrics = calculate_model_performance(lda_model, corpus, dictionary, verbose)
    
    # Step 5: Load original reviews for analysis
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        original_reviews = processed_data['original_reviews']
    except FileNotFoundError:
        if verbose:
            print(f"⚠️ Warning: Could not load original reviews from {processed_file_path}")
        original_reviews = [f"Review {i+1}" for i in range(len(corpus))]
    
    # Step 6: Analyze document topics
    document_topics = analyze_document_topics(
        lda_model, corpus, original_reviews, num_samples, verbose
    )
    
    # Step 7: Calculate topic distribution
    topic_distribution = calculate_topic_distribution(lda_model, corpus, topic_summaries, verbose)
    
    # Step 8: Generate business insights
    business_insights = generate_business_insights(topic_summaries, verbose)
    
    # Step 9: Prepare results for saving
    results_data = {
        'model_parameters': {
            'num_topics': num_topics,
            'passes': passes,
            'alpha': alpha,
            'eta': eta,
            'random_state': random_state
        },
        'model_performance': performance_metrics,
        'topics': convert_numpy_types(topic_summaries),
        'topic_distribution': topic_distribution,
        'sample_document_analysis': convert_numpy_types(document_topics),
        'business_insights': business_insights
    }
    
    # Step 10: Save model and results
    file_paths = save_trained_model(lda_model, results_data, models_dir, verbose)
    
    # Compile complete results
    complete_results = {
        'lda_model': lda_model,
        'dictionary': dictionary,
        'corpus': corpus,
        'metadata': metadata,
        'topic_summaries': topic_summaries,
        'performance_metrics': performance_metrics,
        'document_topics': document_topics,
        'topic_distribution': topic_distribution,
        'business_insights': business_insights,
        'file_paths': file_paths,
        'pipeline_config': {
            'models_dir': models_dir,
            'processed_file_path': processed_file_path,
            'num_topics': num_topics,
            'passes': passes,
            'alpha': alpha,
            'eta': eta,
            'random_state': random_state,
            'num_words': num_words,
            'num_samples': num_samples
        }
    }
    
    if verbose:
        print("\n" + "="*50)
        print("LDA TRAINING PIPELINE COMPLETE!")
        print("="*50)
        print("✅ LDA model successfully trained")
        print("✅ Topics discovered and analyzed")
        print("✅ Model performance metrics calculated")
        print("✅ Topic distribution analyzed")
        print("✅ Model and results saved")
        print("✅ Business insights generated")
    
    return complete_results


if __name__ == "__main__":
    # Run the pipeline if called directly
    results = train_lda_pipeline()
    print(f"\nTraining completed successfully!")
    print(f"Number of topics: {len(results['topic_summaries'])}")
    print(f"Model saved to: {results['file_paths']['model_path']}")
    print(f"Results saved to: {results['file_paths']['results_path']}")
