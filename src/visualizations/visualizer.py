"""
Visualization functions for Market Review LDA analysis.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def load_model_and_results(models_dir="src/models", verbose=True):
    """
    Load trained LDA model, results, dictionary, and corpus.
    
    Args:
        models_dir (str): Directory containing the saved models
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (lda_model, results, dictionary, corpus)
    """
    if verbose:
        print("\n📁 Loading trained model and results...")
    
    try:
        # Load results
        results_path = os.path.join(models_dir, "lda_results.json")
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Load trained model
        model_path = os.path.join(models_dir, "lda_model")
        lda_model = LdaModel.load(model_path)
        
        # Load dictionary and corpus
        dictionary = corpora.Dictionary.load(os.path.join(models_dir, "review_dictionary.dict"))
        corpus = corpora.MmCorpus(os.path.join(models_dir, "review_corpus.mm"))
        
        if verbose:
            print("✅ All model files loaded successfully")
        
        return lda_model, results, dictionary, corpus
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found: {e}. Please run LDA training first.")


def create_topic_distribution_chart(results, output_dir, verbose=True):
    """
    Create and save topic distribution bar chart.
    
    Args:
        results (dict): LDA results dictionary
        output_dir (str): Directory to save the chart
        verbose (bool): Whether to print detailed information
        
    Returns:
        str: Path to saved chart
    """
    if verbose:
        print("\n📊 Creating topic distribution chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    topics = results['topics']
    topic_counts = results['topic_distribution']
    num_topics = results['model_parameters']['num_topics']
    
    # Prepare data
    topic_labels = []
    counts = []
    colors = plt.cm.Set3(np.linspace(0, 1, num_topics))
    
    for i in range(num_topics):
        top_words = ', '.join(topics[i]['top_words'][:3])
        topic_labels.append(f"Topic {i+1}\n({top_words})")
        counts.append(topic_counts[str(i)])
    
    # Create bar chart
    bars = ax.bar(topic_labels, counts, color=colors)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Topic Distribution Across Customer Reviews', fontsize=16, fontweight='bold')
    ax.set_xlabel('Topics', fontsize=12)
    ax.set_ylabel('Number of Reviews', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(output_dir, "topic_distribution.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✅ Topic distribution chart saved to: {chart_path}")
    
    return chart_path


def create_topic_wordclouds(results, output_dir, verbose=True):
    """
    Create and save word clouds for each topic.
    
    Args:
        results (dict): LDA results dictionary
        output_dir (str): Directory to save the word clouds
        verbose (bool): Whether to print detailed information
        
    Returns:
        str: Path to saved word clouds
    """
    if verbose:
        print("\n☁️ Creating word clouds for each topic...")
    
    topics = results['topics']
    num_topics = results['model_parameters']['num_topics']
    
    # Calculate grid size
    cols = min(3, num_topics)
    rows = (num_topics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if num_topics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(num_topics):
        # Get topic words and probabilities
        topic_words = dict(topics[i]['all_words'][:15])  # Top 15 words
        
        # Create word cloud
        wordcloud = WordCloud(
            width=400, 
            height=300,
            background_color='white',
            colormap='viridis',
            max_words=20
        ).generate_from_frequencies(topic_words)
        
        # Plot
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Topic {i+1}: {", ".join(topics[i]["top_words"][:3])}', 
                          fontsize=14, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_topics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    wordcloud_path = os.path.join(output_dir, "topic_wordclouds.png")
    plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✅ Word clouds saved to: {wordcloud_path}")
    
    return wordcloud_path


def create_topic_document_heatmap(lda_model, corpus, results, output_dir, max_docs=20, verbose=True):
    """
    Create and save topic-document heatmap.
    
    Args:
        lda_model: Trained LDA model
        corpus: Corpus data
        results (dict): LDA results dictionary
        output_dir (str): Directory to save the heatmap
        max_docs (int): Maximum number of documents to show
        verbose (bool): Whether to print detailed information
        
    Returns:
        str: Path to saved heatmap
    """
    if verbose:
        print("\n🔥 Creating topic-document heatmap...")
    
    num_topics = results['model_parameters']['num_topics']
    
    # Get topic probabilities for all documents
    doc_topic_probs = []
    for doc_bow in corpus:
        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0.01)
        
        # Create probability array for all topics
        topic_probs = [0.0] * num_topics
        for topic_id, prob in doc_topics:
            topic_probs[topic_id] = prob
        
        doc_topic_probs.append(topic_probs)
    
    # Convert to DataFrame for easier handling
    topic_cols = [f'Topic {i+1}' for i in range(num_topics)]
    heatmap_data = pd.DataFrame(doc_topic_probs, columns=topic_cols)
    
    # Create heatmap (show first max_docs documents)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data.head(max_docs), 
                annot=True, 
                fmt='.2f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Topic Probability'})
    
    plt.title(f'Topic Probability Distribution Across Documents\n(First {max_docs} Reviews)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Topics', fontsize=12)
    plt.ylabel('Review Number', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, "topic_document_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✅ Topic-document heatmap saved to: {heatmap_path}")
    
    return heatmap_path


def generate_marketing_insights(lda_model, corpus, results, processed_file_path, verbose=True):
    """
    Generate comprehensive marketing insights from LDA results.
    
    Args:
        lda_model: Trained LDA model
        corpus: Corpus data
        results (dict): LDA results dictionary
        processed_file_path (str): Path to processed reviews JSON file
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Marketing insights dictionary
    """
    if verbose:
        print("\n💼 Generating marketing insights...")
    
    topics = results['topics']
    topic_counts = results['topic_distribution']
    num_topics = results['model_parameters']['num_topics']
    
    # Load original reviews
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        original_reviews = processed_data['original_reviews']
    except FileNotFoundError:
        if verbose:
            print(f"⚠️ Warning: Could not load original reviews from {processed_file_path}")
        original_reviews = [f"Review {i+1}" for i in range(len(corpus))]
    
    # Create comprehensive analysis
    marketing_insights = {
        'topic_analysis': {},
        'review_classification': [],
        'recommendations': []
    }
    
    # Analyze each topic
    for i, topic in enumerate(topics):
        top_words = topic['top_words'][:5]
        count = topic_counts[str(i)]
        percentage = (count / sum(topic_counts.values())) * 100
        
        # Determine likely business category
        business_category = "General Feedback"
        action_priority = "Medium"
        
        if any(word in ['battery', 'charge', 'power', 'life'] for word in top_words):
            business_category = "Product Performance - Battery"
            action_priority = "High" if percentage > 25 else "Medium"
        elif any(word in ['camera', 'photo', 'picture', 'quality'] for word in top_words):
            business_category = "Product Feature - Camera"
            action_priority = "Medium"
        elif any(word in ['service', 'support', 'help', 'staff'] for word in top_words):
            business_category = "Customer Service"
            action_priority = "High" if percentage > 20 else "Medium"
        elif any(word in ['price', 'money', 'cost', 'expensive', 'value'] for word in top_words):
            business_category = "Pricing & Value"
            action_priority = "High" if percentage > 20 else "Medium"
        elif any(word in ['delivery', 'shipping', 'fast', 'quick'] for word in top_words):
            business_category = "Logistics & Delivery"
            action_priority = "Medium"
        
        marketing_insights['topic_analysis'][f'topic_{i+1}'] = {
            'keywords': top_words,
            'review_count': count,
            'percentage': round(percentage, 1),
            'business_category': business_category,
            'action_priority': action_priority
        }
    
    # Classify all reviews
    for idx, doc_bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_bow)
        if doc_topics:
            dominant_topic_id, dominant_prob = max(doc_topics, key=lambda x: x[1])
            
            marketing_insights['review_classification'].append({
                'review_id': idx + 1,
                'review_text': original_reviews[idx] if idx < len(original_reviews) else "",
                'dominant_topic': dominant_topic_id + 1,
                'topic_probability': round(dominant_prob, 3),
                'business_category': marketing_insights['topic_analysis'][f'topic_{dominant_topic_id+1}']['business_category']
            })
    
    # Generate recommendations
    high_priority_topics = [
        topic for topic, data in marketing_insights['topic_analysis'].items() 
        if data['action_priority'] == 'High'
    ]
    
    if high_priority_topics:
        for topic in high_priority_topics:
            topic_data = marketing_insights['topic_analysis'][topic]
            if 'Battery' in topic_data['business_category']:
                marketing_insights['recommendations'].append({
                    'topic': topic,
                    'category': topic_data['business_category'],
                    'action': 'Improve battery performance or highlight battery features in marketing',
                    'priority': 'High',
                    'impact': f"{topic_data['percentage']}% of reviews mention this"
                })
            elif 'Customer Service' in topic_data['business_category']:
                marketing_insights['recommendations'].append({
                    'topic': topic,
                    'category': topic_data['business_category'],
                    'action': 'Enhance customer service training and response times',
                    'priority': 'High',
                    'impact': f"{topic_data['percentage']}% of reviews mention this"
                })
    
    return marketing_insights


def create_marketing_report(results, marketing_insights, output_dir, verbose=True):
    """
    Create comprehensive marketing report.
    
    Args:
        results (dict): LDA results dictionary
        marketing_insights (dict): Marketing insights dictionary
        output_dir (str): Directory to save the report
        verbose (bool): Whether to print detailed information
        
    Returns:
        str: Path to saved report
    """
    if verbose:
        print("\n📋 Generating marketing report...")
    
    num_topics = results['model_parameters']['num_topics']
    total_reviews = len(marketing_insights['review_classification'])
    
    report_content = f"""
# CUSTOMER REVIEW TOPIC ANALYSIS - MARKETING REPORT

## Executive Summary
- **Total Reviews Analyzed**: {total_reviews}
- **Topics Discovered**: {num_topics}
- **Model Performance**: Log Perplexity = {results['model_performance']['log_perplexity']:.3f}

## Key Findings

"""
    
    for topic_key, topic_data in marketing_insights['topic_analysis'].items():
        report_content += f"""
### {topic_key.replace('_', ' ').title()}: {topic_data['business_category']}
- **Keywords**: {', '.join(topic_data['keywords'])}
- **Coverage**: {topic_data['review_count']} reviews ({topic_data['percentage']}%)
- **Priority**: {topic_data['action_priority']}

"""
    
    if marketing_insights['recommendations']:
        report_content += "\n## Recommended Actions\n\n"
        for rec in marketing_insights['recommendations']:
            report_content += f"""
- **{rec['category']}** ({rec['priority']} Priority)
  - Action: {rec['action']}
  - Impact: {rec['impact']}

"""
    
    report_content += f"""
## Marketing Applications

1. **Segment Customers**: Use topic assignments to create targeted customer segments
2. **Content Strategy**: Create content addressing each major topic area
3. **Product Development**: Focus improvements on high-frequency negative topics
4. **Customer Service**: Train staff on issues highlighted in service-related topics
5. **Marketing Messaging**: Emphasize strengths shown in positive topic areas

## Files Generated
- Topic Distribution Chart: topic_distribution.png
- Topic Word Clouds: topic_wordclouds.png
- Topic-Document Heatmap: topic_document_heatmap.png
- Detailed Insights: marketing_insights.json
"""
    
    report_path = os.path.join(output_dir, "marketing_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    if verbose:
        print(f"✅ Marketing report saved to: {report_path}")
    
    return report_path


def classify_new_review(review_text, lda_model, dictionary, marketing_insights):
    """
    Classify a new review using the trained LDA model.
    
    Args:
        review_text (str): The review text to classify
        lda_model: Trained LDA model
        dictionary: Gensim dictionary
        marketing_insights (dict): Marketing insights dictionary
        
    Returns:
        dict: Classification results
    """
    try:
        # Initialize preprocessing tools
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Clean text (same preprocessing as training)
        text = review_text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                  if word not in stop_words and len(word) > 2]
        
        # Convert to corpus format
        new_doc_bow = dictionary.doc2bow(tokens)
        
        # Get topic probabilities
        doc_topics = lda_model.get_document_topics(new_doc_bow)
        
        if doc_topics:
            dominant_topic_id, dominant_prob = max(doc_topics, key=lambda x: x[1])
            topic_info = marketing_insights['topic_analysis'][f'topic_{dominant_topic_id + 1}']
            
            return {
                'review_text': review_text,
                'dominant_topic': dominant_topic_id + 1,
                'topic_probability': round(dominant_prob, 3),
                'business_category': topic_info['business_category'],
                'keywords': topic_info['keywords'],
                'all_topics': [(tid + 1, round(prob, 3)) for tid, prob in doc_topics]
            }
        else:
            return {
                'review_text': review_text,
                'dominant_topic': None,
                'message': 'Unable to classify - review may be too short or contain unfamiliar words'
            }
    except Exception as e:
        return {
            'review_text': review_text,
            'error': str(e),
            'message': 'Error occurred during classification'
        }


def create_visualizations_pipeline(models_dir="src/models", 
                                   processed_file_path="src/data/processed/processed_reviews.json",
                                   output_dir="src/visualization_results",
                                   max_heatmap_docs=20,
                                   verbose=True):
    """
    Complete pipeline for creating visualizations and marketing insights.
    
    Args:
        models_dir (str): Directory containing the saved models
        processed_file_path (str): Path to processed reviews JSON file
        output_dir (str): Directory to save visualization outputs
        max_heatmap_docs (int): Maximum number of documents to show in heatmap
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Complete visualization results and file paths
    """
    if verbose:
        print("\n" + "="*60)
        print("VISUALIZATION & MARKETING ANALYSIS PIPELINE")
        print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Step 1: Load model and results
    lda_model, results, dictionary, corpus = load_model_and_results(models_dir, verbose)
    
    # Step 2: Create visualizations
    chart_path = create_topic_distribution_chart(results, output_dir, verbose)
    wordcloud_path = create_topic_wordclouds(results, output_dir, verbose)
    heatmap_path = create_topic_document_heatmap(lda_model, corpus, results, output_dir, max_heatmap_docs, verbose)
    
    # Step 3: Generate marketing insights
    marketing_insights = generate_marketing_insights(lda_model, corpus, results, processed_file_path, verbose)
    
    # Step 4: Save marketing insights
    insights_path = os.path.join(output_dir, "marketing_insights.json")
    with open(insights_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(marketing_insights), f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"✅ Marketing insights saved to: {insights_path}")
    
    # Step 5: Create marketing report
    report_path = create_marketing_report(results, marketing_insights, output_dir, verbose)
    
    # Step 6: Test classifier
    if verbose:
        print("\n🧪 Testing new review classifier...")
        test_review = "The battery dies too quickly and customer support was not helpful at all"
        classification_result = classify_new_review(test_review, lda_model, dictionary, marketing_insights)
        print(f"Test Review: '{test_review}'")
        print(f"Classification: {classification_result}")
    
    # Compile results
    visualization_results = {
        'lda_model': lda_model,
        'dictionary': dictionary,
        'corpus': corpus,
        'results': results,
        'marketing_insights': marketing_insights,
        'file_paths': {
            'topic_distribution_chart': chart_path,
            'topic_wordclouds': wordcloud_path,
            'topic_document_heatmap': heatmap_path,
            'marketing_insights': insights_path,
            'marketing_report': report_path,
            'output_directory': output_dir
        },
        'pipeline_config': {
            'models_dir': models_dir,
            'processed_file_path': processed_file_path,
            'output_dir': output_dir,
            'max_heatmap_docs': max_heatmap_docs
        }
    }
    
    if verbose:
        print("\n" + "="*50)
        print("VISUALIZATION PIPELINE COMPLETE!")
        print("="*50)
        print("✅ Topic distribution chart created")
        print("✅ Topic word clouds generated")
        print("✅ Topic-document heatmap created")
        print("✅ Marketing insights generated")
        print("✅ Marketing report created")
        print("✅ Review classifier ready for use")
        print(f"\n📂 All outputs saved to: {output_dir}/")
    
    return visualization_results


if __name__ == "__main__":
    # Run the pipeline if called directly
    results = create_visualizations_pipeline()
    print(f"\nVisualization pipeline completed successfully!")
    print(f"Output directory: {results['file_paths']['output_directory']}")
