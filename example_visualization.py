"""
Example script showing how to use the visualization functions directly.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from visualizations.visualizer import (
    create_visualizations_pipeline,
    create_topic_distribution_chart,
    create_topic_wordclouds,
    create_topic_document_heatmap,
    generate_marketing_insights,
    create_marketing_report,
    classify_new_review,
    load_model_and_results
)


def example_visualization_usage():
    """Example of using the visualization functions."""
    
    print("📊 Example: Using the Visualization Functions")
    print("=" * 50)
    
    # Option 1: Use the complete pipeline
    print("\n🔧 Option 1: Complete Pipeline")
    viz_results = create_visualizations_pipeline(
        models_dir="src/models",
        processed_file_path="src/data/processed/processed_reviews.json",
        output_dir="src/visualization_results",
        verbose=True
    )
    
    print(f"✅ Pipeline completed. Files saved to: {viz_results['file_paths']['output_directory']}")
    
    # Option 2: Use individual functions
    print("\n🔧 Option 2: Individual Functions")
    
    # Load model and results first
    lda_model, results, dictionary, corpus = load_model_and_results("src/models", verbose=False)
    
    # Create individual visualizations
    output_dir = "src/custom_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create just the topic distribution chart
    chart_path = create_topic_distribution_chart(results, output_dir, verbose=False)
    print(f"✅ Topic chart: {chart_path}")
    
    # Create just the word clouds
    wordcloud_path = create_topic_wordclouds(results, output_dir, verbose=False)
    print(f"✅ Word clouds: {wordcloud_path}")
    
    # Test the new review classifier
    print("\n🧪 Testing New Review Classifier:")
    
    # Generate marketing insights first (needed for classifier)
    marketing_insights = generate_marketing_insights(
        lda_model, corpus, results, 
        "src/data/processed/processed_reviews.json", 
        verbose=False
    )
    
    # Test classification
    test_reviews = [
        "The battery life is terrible and drains very fast",
        "Great camera quality and excellent photos",
        "Customer service was unhelpful and rude"
    ]
    
    for review in test_reviews:
        result = classify_new_review(review, lda_model, dictionary, marketing_insights)
        print(f"\nReview: '{review}'")
        if 'dominant_topic' in result and result['dominant_topic']:
            print(f"  → Topic {result['dominant_topic']}: {result['business_category']}")
            print(f"  → Probability: {result['topic_probability']:.3f}")
            print(f"  → Keywords: {result['keywords'][:3]}")
        else:
            print(f"  → {result.get('message', 'Classification failed')}")
    
    return viz_results


if __name__ == "__main__":
    try:
        results = example_visualization_usage()
        print(f"\n✅ Visualization example completed successfully!")
        print(f"📂 Check the output directories for generated files:")
        print(f"   - Main results: {results['file_paths']['output_directory']}")
        print(f"   - Custom files: src/custom_visualizations/")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have run the LDA training pipeline first!")
