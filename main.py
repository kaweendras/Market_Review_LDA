"""
Example script showing how to use the modular functions directly.
"""

import sys
import os
import subprocess
import platform

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.preprocessing import process_reviews_pipeline
from data.corpus_builder import create_dictionary_and_corpus_pipeline
from training.lda_trainer import train_lda_pipeline
from visualizations.visualizer import create_visualizations_pipeline


def get_user_confirmation(step_name, description):
    """Get user confirmation before proceeding with a step."""
    print(f"\n{'='*60}")
    print(f"Ready to start: {step_name}")
    print(f"Description: {description}")
    print(f"{'='*60}")
    
    while True:
        response = input("Do you want to proceed? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            print("❌ Step cancelled by user.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def open_file_with_system(file_path):
    """Open a file using the system's default application."""
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_path])
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", file_path])
        return True
    except Exception as e:
        print(f"❌ Error opening file {file_path}: {e}")
        return False


def open_visualization_files(viz_results):
    """Ask user if they want to open the generated visualization files."""
    if not viz_results or 'file_paths' not in viz_results:
        print("❌ No visualization files to open.")
        return
    
    print(f"\n{'='*60}")
    print("📂 Generated Files Available:")
    print(f"{'='*60}")
    
    # List available files
    file_paths = viz_results['file_paths']
    available_files = []
    
    # Check for marketing report
    if 'marketing_report' in file_paths:
        report_path = file_paths['marketing_report']
        print(f"🔍 Checking marketing report: {report_path}")
        if os.path.exists(report_path):
            available_files.append(('Marketing Report (MD)', report_path))
            print(f"✅ Marketing Report: {report_path}")
        else:
            print(f"❌ Marketing report not found at: {report_path}")
    
    # Check for visualization files in output directory
    if 'output_directory' in file_paths:
        output_dir = file_paths['output_directory']
        print(f"🔍 Checking output directory: {output_dir}")
        
        if os.path.exists(output_dir):
            print(f"✅ Output directory exists, scanning for files...")
            
            # List all files in the directory
            all_files = os.listdir(output_dir)
            print(f"📁 Found {len(all_files)} files in directory: {all_files}")
            
            # Look for image files and other visualization files
            for filename in all_files:
                file_path = os.path.join(output_dir, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
                    available_files.append(('Image Visualization', file_path))
                    print(f"🖼️  Image: {file_path}")
                elif filename.lower().endswith(('.html', '.htm')):
                    available_files.append(('HTML Visualization', file_path))
                    print(f"🌐 HTML: {file_path}")
                elif filename.lower().endswith('.md'):
                    available_files.append(('Markdown File', file_path))
                    print(f"📄 Markdown: {file_path}")
        else:
            print(f"❌ Output directory not found: {output_dir}")
    
    if not available_files:
        print("❌ No files found to open.")
        print("🔍 Debug info:")
        print(f"   - viz_results keys: {list(viz_results.keys()) if viz_results else 'None'}")
        if viz_results and 'file_paths' in viz_results:
            print(f"   - file_paths keys: {list(viz_results['file_paths'].keys())}")
        return
    
    print(f"\n📊 Found {len(available_files)} files to open.")
    
    while True:
        response = input("\nDo you want to open these files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n🔄 Opening files...")
            opened_count = 0
            for file_type, file_path in available_files:
                print(f"Opening {file_type}: {os.path.basename(file_path)}")
                if open_file_with_system(file_path):
                    opened_count += 1
                else:
                    print(f"❌ Failed to open: {file_path}")
                    
            print(f"✅ Successfully opened {opened_count}/{len(available_files)} files")
            break
        elif response in ['n', 'no']:
            print("📁 Files remain available in the output directory for manual viewing.")
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def example_usage():
    """Example of using the modular functions."""
    
    print("🔧 Example: Using the Market Review LDA Analysis functions")
    print("=" * 60)
    print("This script will guide you through 4 steps of the LDA analysis pipeline.")
    print("You'll be asked for confirmation before each step begins.")
    
    # Step 1: Run preprocessing
    if not get_user_confirmation(
        "Step 1: Text Preprocessing", 
        "Process and clean the raw review text data"
    ):
        return None, None, None, None
    
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
    if not get_user_confirmation(
        "Step 2: Dictionary and Corpus Creation",
        "Create dictionary and corpus from processed text data"
    ):
        return preprocessing_results, None, None, None
    
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
    if not get_user_confirmation(
        "Step 3: LDA Model Training",
        "Train the LDA model on the processed corpus"
    ):
        return preprocessing_results, corpus_results, None, None
    
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
    if not get_user_confirmation(
        "Step 4: Visualizations & Marketing Insights",
        "Generate visualizations and extract marketing insights from the trained model"
    ):
        return preprocessing_results, corpus_results, lda_results, None
    
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
    
    # Optionally, open the visualization files
    open_visualization_files(viz_results)
