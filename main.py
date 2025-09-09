"""
Example script showing how to use the modular functions directly.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.preprocessing import process_reviews_pipeline
from data.corpus_builder import create_dictionary_and_corpus_pipeline
from training.lda_trainer import train_lda_pipeline
from visualizations.visualizer import create_visualizations_pipeline


def example_usage():
    """Example of using the modular functions."""
    
    print("🔧 Example: Using the Market Review LDA Analysis functions")
    print("=" * 60)
    
    # Step 1: Run preprocessing
    print("\n📝 Step 1: Text Preprocessing")
    preprocessing_results = process_reviews_pipeline(
        csv_path="src/data/raw/sample_reviews.csv",
        output_dir="src/data/processed",
        top_words=5,
        verbose=False
    )
    
    print(f"✅ Processed {len(preprocessing_results['processed_reviews'])} reviews")
    print(f"📊 Found {preprocessing_results['word_analysis']['total_unique_words']} unique words")
    
    # Step 2: Create dictionary and corpus
    print("\n📚 Step 2: Dictionary and Corpus Creation")
    corpus_results = create_dictionary_and_corpus_pipeline(
        processed_file_path=preprocessing_results['file_paths']['json_path'],
        models_dir="src/models",
        no_below=1,  # Less restrictive for small dataset
        no_above=0.9,
        keep_n=100,
        verbose=False
    )
    
    print(f"✅ Dictionary size: {len(corpus_results['dictionary'])}")
    print(f"✅ Corpus size: {len(corpus_results['corpus'])}")
    
    # Step 3: Train LDA model
    print("\n🤖 Step 3: LDA Model Training")
    lda_results = train_lda_pipeline(
        models_dir="src/models",
        processed_file_path=preprocessing_results['file_paths']['json_path'],
        num_topics=3,  # Using 3 topics for small dataset
        passes=10,
        alpha='auto',
        eta='auto',
        random_state=42,
        num_words=5,
        num_samples=3,
        verbose=False
    )
    
    print(f"✅ LDA model trained with {len(lda_results['topic_summaries'])} topics")
    print(f"✅ Model performance: {lda_results['performance_metrics']['log_perplexity']:.2f} log perplexity")
    
    # Step 4: Create visualizations and marketing insights
    print("\n📊 Step 4: Visualizations & Marketing Insights")
    viz_results = create_visualizations_pipeline(
        models_dir="src/models",
        processed_file_path=preprocessing_results['file_paths']['json_path'],
        output_dir="src/visualization_results",
        max_heatmap_docs=15,
        verbose=False
    )
    
    print(f"✅ Visualizations created: {len(viz_results['file_paths'])} files")
    print(f"✅ Marketing insights generated for {len(viz_results['marketing_insights']['topic_analysis'])} topics")
    
    # Show how to access the results
    print(f"\n📋 Example results access:")
    print(f"Top 3 words: {preprocessing_results['word_analysis']['top_words'][:3]}")
    print(f"Dictionary keys: {list(corpus_results['dictionary'].token2id.keys())[:5]}")
    print(f"First document in corpus: {corpus_results['corpus'][0]}")
    print(f"Topic 0 top words: {lda_results['topic_summaries'][0]['top_words']}")
    print(f"Visualization files: {list(viz_results['file_paths'].keys())[:3]}")
    
    return preprocessing_results, corpus_results, lda_results, viz_results


if __name__ == "__main__":
    preprocessing_results, corpus_results, lda_results, viz_results = example_usage()
    print(f"\n✅ Example completed successfully!")
    print(f"📊 Final Summary:")
    print(f"   - Processed {len(preprocessing_results['processed_reviews'])} reviews")
    print(f"   - Created dictionary with {len(corpus_results['dictionary'])} words")
    print(f"   - Trained LDA model with {len(lda_results['topic_summaries'])} topics")
    print(f"   - Generated {len(viz_results['file_paths'])} visualization files")
    print(f"   - Model saved to: {lda_results['file_paths']['model_path']}")
    print(f"   - Results saved to: {lda_results['file_paths']['results_path']}")
    print(f"   - Visualizations saved to: {viz_results['file_paths']['output_directory']}")
    print(f"\n🚀 NEXT STEPS:")
    print(f"   - Review marketing report: {viz_results['file_paths']['marketing_report']}")
    print(f"   - Check visualizations in: {viz_results['file_paths']['output_directory']}")
    print(f"   - Use the classification pipeline for new reviews")
