import json
import os
from gensim import corpora
from collections import Counter
import pandas as pd


def load_processed_reviews(processed_file_path="src/data/processed/processed_reviews.json"):
    """
    Load processed reviews from JSON file.
    
    Args:
        processed_file_path (str): Path to the processed reviews JSON file
        
    Returns:
        tuple: (processed_reviews, original_reviews, stats)
    """
    print(f"📁 Loading processed reviews from: {processed_file_path}")
    
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        processed_reviews = processed_data['processed_reviews']
        original_reviews = processed_data['original_reviews']
        stats = processed_data['preprocessing_stats']
        
        print(f"✅ Loaded {stats['total_reviews']} processed reviews")
        print(f"📈 Total unique words: {stats['total_unique_words']}")
        print(f"📈 Total words: {stats['total_words']}")
        
        return processed_reviews, original_reviews, stats
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Processed data file not found at {processed_file_path}. Please run preprocessing first.")
    except KeyError as e:
        raise KeyError(f"Invalid processed data format. Missing key: {e}")


def create_dictionary(processed_reviews, verbose=True):
    """
    Create Gensim dictionary from processed reviews.
    
    Args:
        processed_reviews (list): List of processed token lists
        verbose (bool): Whether to print detailed information
        
    Returns:
        gensim.corpora.Dictionary: Dictionary object mapping words to IDs
    """
    if verbose:
        print("\n🔤 Creating Gensim Dictionary...")
        print("This maps each unique word to a unique ID number")
    
    # Create dictionary from processed reviews
    dictionary = corpora.Dictionary(processed_reviews)
    
    if verbose:
        print(f"✅ Dictionary created with {len(dictionary)} unique words")
        
        # Show sample word-to-ID mappings
        print("\n📝 Sample word-to-ID mappings:")
        sample_words = list(dictionary.token2id.items())[:10]
        for word, word_id in sample_words:
            print(f"  Word: '{word}' → ID: {word_id}")
    
    return dictionary


def filter_dictionary(dictionary, no_below=2, no_above=0.8, keep_n=1000, verbose=True):
    """
    Filter dictionary to remove words that appear too rarely or too frequently.
    
    Args:
        dictionary (gensim.corpora.Dictionary): Dictionary to filter
        no_below (int): Ignore words that appear in less than this many documents
        no_above (float): Ignore words that appear in more than this fraction of documents
        keep_n (int): Keep only the n most frequent words
        verbose (bool): Whether to print detailed information
        
    Returns:
        gensim.corpora.Dictionary: Filtered dictionary
    """
    if verbose:
        print("\n🔍 Filtering dictionary...")
        print("Removing words that appear in very few or too many documents")
        print(f"Dictionary size before filtering: {len(dictionary)}")
    
    # Filter extremes
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    
    if verbose:
        print(f"Dictionary size after filtering: {len(dictionary)}")
        print(f"  - Removed words appearing in < {no_below} documents")
        print(f"  - Removed words appearing in > {no_above*100}% of documents")
        print(f"  - Kept only top {keep_n} most frequent words")
    
    return dictionary


def create_corpus(processed_reviews, dictionary, verbose=True):
    """
    Create corpus (bag of words representation) from processed reviews.
    
    Args:
        processed_reviews (list): List of processed token lists
        dictionary (gensim.corpora.Dictionary): Dictionary for word-to-ID mapping
        verbose (bool): Whether to print detailed information
        
    Returns:
        list: Corpus as list of lists containing (word_id, frequency) tuples
    """
    if verbose:
        print("\n📊 Creating Corpus (Bag of Words representation)...")
        print("Converting each review into (word_id, frequency) pairs")
    
    # Create corpus - each document becomes a list of (word_id, word_frequency) tuples
    corpus = [dictionary.doc2bow(review) for review in processed_reviews]
    
    if verbose:
        print(f"✅ Corpus created with {len(corpus)} documents")
    
    return corpus


def analyze_corpus_sample(corpus, dictionary, original_reviews, processed_reviews, doc_index=0, verbose=True):
    """
    Analyze and display a sample document from the corpus.
    
    Args:
        corpus (list): Corpus data
        dictionary (gensim.corpora.Dictionary): Dictionary for word mapping
        original_reviews (list): Original review texts
        processed_reviews (list): Processed token lists
        doc_index (int): Index of document to analyze
        verbose (bool): Whether to print detailed information
    """
    if not verbose:
        return
        
    print("\n🔍 Sample corpus document analysis:")
    print(f"Original review: '{original_reviews[doc_index][:100]}...'")
    print(f"Processed tokens: {processed_reviews[doc_index]}")
    print(f"Corpus representation: {corpus[doc_index]}")
    
    print("\nExplaining the corpus format:")
    for word_id, frequency in corpus[doc_index][:5]:  # Show first 5 word-frequency pairs
        word = dictionary[word_id]
        print(f"  Word ID {word_id} ('{word}') appears {frequency} times")


def calculate_corpus_statistics(corpus, dictionary, original_reviews, verbose=True):
    """
    Calculate and display corpus statistics.
    
    Args:
        corpus (list): Corpus data
        dictionary (gensim.corpora.Dictionary): Dictionary object
        original_reviews (list): Original review texts
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Dictionary containing corpus statistics
    """
    # Calculate corpus statistics
    total_word_occurrences = sum(sum(freq for _, freq in doc) for doc in corpus)
    unique_words_in_corpus = len(dictionary)
    avg_words_per_doc = total_word_occurrences / len(corpus) if len(corpus) > 0 else 0
    
    # Find documents with most/least words
    doc_lengths = [sum(freq for _, freq in doc) for doc in corpus]
    max_length_idx = doc_lengths.index(max(doc_lengths)) if doc_lengths else 0
    min_length_idx = doc_lengths.index(min(doc_lengths)) if doc_lengths else 0
    
    stats = {
        'total_word_occurrences': total_word_occurrences,
        'unique_words_in_corpus': unique_words_in_corpus,
        'avg_words_per_doc': avg_words_per_doc,
        'max_doc_length': max(doc_lengths) if doc_lengths else 0,
        'min_doc_length': min(doc_lengths) if doc_lengths else 0,
        'max_length_idx': max_length_idx,
        'min_length_idx': min_length_idx,
        'doc_lengths': doc_lengths
    }
    
    if verbose:
        print("\n📈 Corpus Statistics:")
        print(f"📊 Total word occurrences across all documents: {total_word_occurrences}")
        print(f"📊 Unique words in corpus: {unique_words_in_corpus}")
        print(f"📊 Average words per document: {avg_words_per_doc:.2f}")
        
        print(f"\n📏 Document length analysis:")
        print(f"Longest document: #{max_length_idx + 1} with {stats['max_doc_length']} words")
        print(f"  Text: '{original_reviews[max_length_idx][:100]}...'")
        print(f"Shortest document: #{min_length_idx + 1} with {stats['min_doc_length']} words")
        print(f"  Text: '{original_reviews[min_length_idx][:100]}...'")
    
    return stats


def save_dictionary_and_corpus(dictionary, corpus, corpus_stats, models_dir="src/models", verbose=True):
    """
    Save dictionary and corpus to files.
    
    Args:
        dictionary (gensim.corpora.Dictionary): Dictionary to save
        corpus (list): Corpus to save
        corpus_stats (dict): Corpus statistics
        models_dir (str): Directory to save models
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Dictionary containing file paths
    """
    if verbose:
        print(f"\n💾 Saving Dictionary and Corpus to: {models_dir}")
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Save dictionary
    dictionary_path = os.path.join(models_dir, "review_dictionary.dict")
    dictionary.save(dictionary_path)
    if verbose:
        print(f"✅ Dictionary saved to: {dictionary_path}")
    
    # Save corpus
    corpus_path = os.path.join(models_dir, "review_corpus.mm")
    corpora.MmCorpus.serialize(corpus_path, corpus)
    if verbose:
        print(f"✅ Corpus saved to: {corpus_path}")
    
    # Save metadata for easy reference later
    metadata = {
        'dictionary_size': len(dictionary),
        'corpus_size': len(corpus),
        'total_word_occurrences': corpus_stats['total_word_occurrences'],
        'avg_words_per_doc': corpus_stats['avg_words_per_doc'],
        'max_doc_length': corpus_stats['max_doc_length'],
        'min_doc_length': corpus_stats['min_doc_length'],
        'dictionary_path': dictionary_path,
        'corpus_path': corpus_path
    }
    
    metadata_path = os.path.join(models_dir, "corpus_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"✅ Metadata saved to: {metadata_path}")
    
    return {
        'dictionary_path': dictionary_path,
        'corpus_path': corpus_path,
        'metadata_path': metadata_path
    }


def validate_corpus(corpus, dictionary, verbose=True):
    """
    Validate the corpus and dictionary for LDA readiness.
    
    Args:
        corpus (list): Corpus to validate
        dictionary (gensim.corpora.Dictionary): Dictionary to validate
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Validation results
    """
    validation_results = {}
    
    # Check if any documents became empty after filtering
    empty_docs = [i for i, doc in enumerate(corpus) if len(doc) == 0]
    validation_results['empty_docs'] = empty_docs
    validation_results['has_empty_docs'] = len(empty_docs) > 0
    
    # Check dictionary consistency
    validation_results['dictionary_size'] = len(dictionary)
    validation_results['corpus_size'] = len(corpus)
    
    if verbose:
        print("\n✅ Validation Checks:")
        
        if validation_results['has_empty_docs']:
            print(f"⚠️ Warning: {len(empty_docs)} documents became empty after filtering")
            print(f"Empty document indices: {empty_docs[:10]}...")  # Show first 10
        else:
            print("✅ No empty documents - corpus is ready for LDA!")
        
        print(f"✅ Dictionary consistency check: {validation_results['dictionary_size']} words mapped")
    
    return validation_results


def analyze_word_frequencies(corpus, dictionary, top_n=10, verbose=True):
    """
    Analyze word frequencies in the corpus.
    
    Args:
        corpus (list): Corpus data
        dictionary (gensim.corpora.Dictionary): Dictionary for word mapping
        top_n (int): Number of top words to show
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Word frequency data
    """
    # Calculate word frequencies
    word_frequencies = {}
    for doc in corpus:
        for word_id, freq in doc:
            if word_id in word_frequencies:
                word_frequencies[word_id] += freq
            else:
                word_frequencies[word_id] = freq
    
    # Sort by frequency
    sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
    top_words = [(dictionary[word_id], freq) for word_id, freq in sorted_words[:top_n]]
    
    if verbose:
        print(f"\n📊 Top {top_n} most frequent words in corpus:")
        for word, total_freq in top_words:
            print(f"  '{word}': {total_freq} total occurrences")
    
    return {
        'word_frequencies': word_frequencies,
        'sorted_words': sorted_words,
        'top_words': top_words
    }


def create_dictionary_and_corpus_pipeline(processed_file_path="src/data/processed/processed_reviews.json",
                                        models_dir="src/models",
                                        no_below=2,
                                        no_above=0.8,
                                        keep_n=1000,
                                        verbose=True):
    """
    Complete pipeline for creating dictionary and corpus from processed reviews.
    
    Args:
        processed_file_path (str): Path to processed reviews JSON file
        models_dir (str): Directory to save models
        no_below (int): Dictionary filtering parameter
        no_above (float): Dictionary filtering parameter
        keep_n (int): Dictionary filtering parameter
        verbose (bool): Whether to print detailed progress
        
    Returns:
        dict: Complete results including dictionary, corpus, and statistics
    """
    if verbose:
        print("\n" + "="*60)
        print("STARTING DICTIONARY AND CORPUS CREATION PIPELINE")
        print("="*60)
    
    # Step 1: Load processed reviews
    processed_reviews, original_reviews, preprocessing_stats = load_processed_reviews(processed_file_path)
    
    # Step 2: Create dictionary
    dictionary = create_dictionary(processed_reviews, verbose)
    
    # Step 3: Filter dictionary
    dictionary = filter_dictionary(dictionary, no_below, no_above, keep_n, verbose)
    
    # Step 4: Create corpus
    corpus = create_corpus(processed_reviews, dictionary, verbose)
    
    # Step 5: Analyze sample document
    if len(corpus) > 0:
        analyze_corpus_sample(corpus, dictionary, original_reviews, processed_reviews, 0, verbose)
    
    # Step 6: Calculate corpus statistics
    corpus_stats = calculate_corpus_statistics(corpus, dictionary, original_reviews, verbose)
    
    # Step 7: Save dictionary and corpus
    file_paths = save_dictionary_and_corpus(dictionary, corpus, corpus_stats, models_dir, verbose)
    
    # Step 8: Validate corpus
    validation_results = validate_corpus(corpus, dictionary, verbose)
    
    # Step 9: Analyze word frequencies
    word_freq_data = analyze_word_frequencies(corpus, dictionary, 10, verbose)
    
    # Compile results
    results = {
        'dictionary': dictionary,
        'corpus': corpus,
        'processed_reviews': processed_reviews,
        'original_reviews': original_reviews,
        'preprocessing_stats': preprocessing_stats,
        'corpus_stats': corpus_stats,
        'validation_results': validation_results,
        'word_frequencies': word_freq_data,
        'file_paths': file_paths,
        'pipeline_config': {
            'processed_file_path': processed_file_path,
            'models_dir': models_dir,
            'no_below': no_below,
            'no_above': no_above,
            'keep_n': keep_n
        }
    }
    
    if verbose:
        print("\n" + "="*50)
        print("DICTIONARY AND CORPUS PIPELINE COMPLETE!")
        print("="*50)
        print("✅ Dictionary created and filtered")
        print("✅ Corpus (Bag of Words) generated")
        print("✅ Dictionary and Corpus saved to models directory")
        print("✅ Validation checks completed")
        print("✅ Data ready for LDA training!")
    
    return results


if __name__ == "__main__":
    # Run the pipeline if called directly
    results = create_dictionary_and_corpus_pipeline()
    print(f"\nPipeline completed successfully!")
    print(f"Dictionary size: {len(results['dictionary'])}")
    print(f"Corpus size: {len(results['corpus'])}")
    print(f"Files saved to: {results['file_paths']}")
