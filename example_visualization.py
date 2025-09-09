import sys
import os
import subprocess
import platform

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


def open_generated_images(output_dir):
    """Open generated images using the default system image viewer."""
    
    # List of image files to look for
    image_files = [
        "topic_distribution.png",
        "topic_wordclouds.png", 
        "topic_document_heatmap.png"
    ]
    
    opened_files = []
    
    for image_file in image_files:
        image_path = os.path.join(output_dir, image_file)
        if os.path.exists(image_path):
            try:
                # Cross-platform way to open files
                if platform.system() == "Windows":
                    os.startfile(image_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", image_path])
                else:  # Linux and others
                    subprocess.run(["xdg-open", image_path])
                opened_files.append(image_file)
                print(f"📸 Opened: {image_file}")
            except Exception as e:
                print(f"❌ Could not open {image_file}: {e}")
        else:
            print(f"⚠️  Image not found: {image_file}")
    
    return opened_files


if __name__ == "__main__":
    try:
        results = example_visualization_usage()
        print(f"\n✅ Visualization example completed successfully!")
        print(f"📂 Check the output directories for generated files:")
        print(f"   - Main results: {results['file_paths']['output_directory']}")
        print(f"   - Custom files: src/custom_visualizations/")
        
        # Open generated images automatically
        print(f"\n🖼️  Opening generated images...")
        opened_files = open_generated_images(results['file_paths']['output_directory'])
        
        if opened_files:
            print(f"✅ Successfully opened {len(opened_files)} image(s)")
        else:
            print(f"⚠️  No images were opened. Check if files exist in {results['file_paths']['output_directory']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have run the LDA training pipeline first!")
