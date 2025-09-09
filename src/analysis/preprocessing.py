import re
import pandas as pd
import os
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk


def setup_nltk():
    print("🔧 Setting up NLTK dependencies...")
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True) 
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️ Error downloading NLTK data: {e}")
        return False


def initialize_preprocessing_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    print("🧹 Setting up text preprocessing tools...")
    print(f"✅ Loaded {len(stop_words)} English stop words")
    print("✅ Initialized lemmatizer")
    
    return lemmatizer, stop_words


def preprocess_text(text, lemmatizer, stop_words):
    """
    Clean and preprocess review text for LDA analysis.
    
    Args:
        text (str): Raw review text
        lemmatizer: NLTK WordNetLemmatizer instance
        stop_words (set): Set of stopwords to remove
        
    Returns:
        list: List of preprocessed tokens
    """
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


def load_reviews_from_csv(csv_path):
    """
    Load reviews from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing reviews
        
    Returns:
        list: List of review texts
    """
    print(f"📁 Loading reviews from CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    reviews = df['review_text'].tolist()
    print(f"✅ Loaded {len(reviews)} reviews")
    return reviews


def analyze_word_frequency(processed_reviews, top_n=10):
    # Flatten all words from all reviews
    all_words = [word for review in processed_reviews for word in review]
    word_counts = Counter(all_words)
    
    print(f"\n📊 Most common words after preprocessing (top {top_n}):")
    top_words = word_counts.most_common(top_n)
    for word, count in top_words:
        print(f"  '{word}': {count} times")
    
    analysis = {
        'word_counts': dict(word_counts),
        'top_words': top_words,
        'total_unique_words': len(word_counts),
        'total_words': len(all_words)
    }
    
    print(f"\n📈 Total unique words: {analysis['total_unique_words']}")
    print(f"📈 Total words: {analysis['total_words']}")
    
    return analysis


def save_processed_data(processed_reviews, original_reviews, word_analysis, output_dir="src/data/processed"):
    """
    Save processed data to files.
    
    Args:
        processed_reviews (list): List of processed token lists
        original_reviews (list): List of original review texts
        word_analysis (dict): Word frequency analysis results
        output_dir (str): Directory to save processed data
        
    Returns:
        dict: Paths of saved files
    """
    print("\n💾 Saving processed data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare processed data
    processed_data = {
        'processed_reviews': processed_reviews,
        'original_reviews': original_reviews,
        'preprocessing_stats': {
            'total_reviews': len(processed_reviews),
            'total_unique_words': word_analysis['total_unique_words'],
            'total_words': word_analysis['total_words']
        },
        'word_frequency': word_analysis['word_counts']
    }
    
    # Save as JSON
    json_path = os.path.join(output_dir, "processed_reviews.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Processed data saved to: {json_path}")
    
    # Save as CSV for easy viewing
    processed_df = pd.DataFrame({
        'review_id': range(1, len(original_reviews) + 1),
        'original_text': original_reviews,
        'processed_tokens': [' '.join(tokens) for tokens in processed_reviews],
        'token_count': [len(tokens) for tokens in processed_reviews]
    })
    
    csv_path = os.path.join(output_dir, "processed_reviews.csv")
    processed_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ Processed data also saved as CSV: {csv_path}")
    
    return {
        'json_path': json_path,
        'csv_path': csv_path
    }


def check_data_quality(processed_reviews):
    """
    Check the quality of processed data.
    
    Args:
        processed_reviews (list): List of processed token lists
        
    Returns:
        dict: Data quality report
    """
    # Check for empty reviews
    empty_reviews = [i for i, review in enumerate(processed_reviews) if len(review) == 0]
    
    if empty_reviews:
        print(f"⚠️ Warning: {len(empty_reviews)} empty reviews after preprocessing")
        print(f"   Empty review indices: {empty_reviews}")
    else:
        print("✅ No empty reviews - good!")
    
    # Additional quality checks
    min_tokens = min(len(review) for review in processed_reviews) if processed_reviews else 0
    max_tokens = max(len(review) for review in processed_reviews) if processed_reviews else 0
    avg_tokens = sum(len(review) for review in processed_reviews) / len(processed_reviews) if processed_reviews else 0
    
    quality_report = {
        'empty_reviews_count': len(empty_reviews),
        'empty_review_indices': empty_reviews,
        'min_tokens_per_review': min_tokens,
        'max_tokens_per_review': max_tokens,
        'avg_tokens_per_review': avg_tokens
    }
    
    print(f"📊 Token statistics: min={min_tokens}, max={max_tokens}, avg={avg_tokens:.1f}")
    
    return quality_report


def process_reviews_pipeline(csv_path="src/data/raw/sample_reviews.csv", 
                           output_dir="src/data/processed",
                           top_words=10,
                           verbose=True):
    """
    Complete pipeline for processing reviews from CSV to analyzed data.
    
    Args:
        csv_path (str): Path to input CSV file
        output_dir (str): Directory to save processed data
        top_words (int): Number of top words to analyze
        verbose (bool): Whether to print detailed progress
        
    Returns:
        dict: Complete analysis results and file paths
    """
    if verbose:
        print("\n" + "="*60)
        print("STARTING REVIEW ANALYSIS PIPELINE")
        print("="*60)
    
    # Step 1: Setup NLTK
    if not setup_nltk():
        raise RuntimeError("Failed to setup NLTK dependencies")
    
    # Step 2: Initialize preprocessing tools
    lemmatizer, stop_words = initialize_preprocessing_tools()
    
    if verbose:
        print(f"\nSample stop words: {list(stop_words)[:20]}...")
    
    # Step 3: Load reviews
    original_reviews = load_reviews_from_csv(csv_path)
    
    # Step 4: Process reviews
    if verbose:
        print(f"\n🔄 Processing {len(original_reviews)} reviews...")
    
    processed_reviews = []
    for i, review in enumerate(original_reviews):
        processed = preprocess_text(review, lemmatizer, stop_words)
        processed_reviews.append(processed)
        if verbose:
            print(f"Review {i+1}: {processed}")
    
    if verbose:
        print(f"\n✅ Processed {len(processed_reviews)} reviews!")
    
    # Step 5: Analyze word frequency
    word_analysis = analyze_word_frequency(processed_reviews, top_words)
    
    # Step 6: Check data quality
    quality_report = check_data_quality(processed_reviews)
    
    # Step 7: Save processed data
    file_paths = save_processed_data(processed_reviews, original_reviews, word_analysis, output_dir)
    
    # Compile results
    results = {
        'processed_reviews': processed_reviews,
        'original_reviews': original_reviews,
        'word_analysis': word_analysis,
        'quality_report': quality_report,
        'file_paths': file_paths,
        'pipeline_config': {
            'csv_path': csv_path,
            'output_dir': output_dir,
            'top_words': top_words
        }
    }
    
    if verbose:
        print("\n" + "="*50)
        print("ANALYSIS PIPELINE COMPLETE!")
        print("="*50)
        print("✅ Text preprocessing completed")
        print("✅ Reviews processed and cleaned")
        print("✅ Word frequency analysis completed")
        print("✅ Data quality checked")
        print("✅ Processed data saved")
        print("✅ Ready for LDA modeling")
    
    return results


if __name__ == "__main__":
    # Run the pipeline if called directly
    results = process_reviews_pipeline()
    print(f"\nPipeline completed successfully!")
    print(f"Processed {len(results['processed_reviews'])} reviews")
    print(f"Files saved to: {results['file_paths']}")
