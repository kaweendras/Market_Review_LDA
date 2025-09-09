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

print("\n" + "="*60)
print("STEP 5: VISUALIZATION & MARKETING APPLICATION")
print("="*60)

# Install required packages if not already installed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    print("✅ Visualization libraries loaded successfully")
except ImportError as e:
    print(f"❌ Please install missing libraries: pip install matplotlib seaborn wordcloud")
    print(f"Error: {e}")
    exit()

# Step 5.1: Load Trained Model and Results
print("\n📁 Loading trained model and results...")

models_dir = "src/models"
results_path = os.path.join(models_dir, "lda_results.json")
model_path = os.path.join(models_dir, "lda_model")

try:
    # Load results
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Load trained model
    lda_model = LdaModel.load(model_path)
    
    # Load dictionary and corpus for additional analysis
    dictionary = corpora.Dictionary.load(os.path.join(models_dir, "review_dictionary.dict"))
    corpus = corpora.MmCorpus(os.path.join(models_dir, "review_corpus.mm"))
    
    print("✅ All model files loaded successfully")
    
except FileNotFoundError as e:
    print(f"❌ Model files not found: {e}")
    print("Please run Step 4 first to train the model.")
    exit()

# Step 5.2: Create Visualizations Directory
print("\n📊 Setting up visualizations...")

viz_dir = "src/visualizations"
os.makedirs(viz_dir, exist_ok=True)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Step 5.3: Topic Distribution Bar Chart
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
chart_path = os.path.join(viz_dir, "topic_distribution.png")
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Topic distribution chart saved to: {chart_path}")

# Step 5.4: Word Clouds for Each Topic
print("\n☁️ Creating word clouds for each topic...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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

# Hide the last subplot if we have fewer topics than subplots
if num_topics < len(axes):
    axes[num_topics].axis('off')

plt.tight_layout()
wordcloud_path = os.path.join(viz_dir, "topic_wordclouds.png")
plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Word clouds saved to: {wordcloud_path}")

# Step 5.5: Topic-Document Heatmap
print("\n🔥 Creating topic-document heatmap...")

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

# Create heatmap (show first 20 documents)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data.head(20), 
            annot=True, 
            fmt='.2f', 
            cmap='YlOrRd',
            cbar_kws={'label': 'Topic Probability'})

plt.title('Topic Probability Distribution Across Documents\n(First 20 Reviews)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Topics', fontsize=12)
plt.ylabel('Review Number', fontsize=12)
plt.tight_layout()

heatmap_path = os.path.join(viz_dir, "topic_document_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Topic-document heatmap saved to: {heatmap_path}")

# Step 5.6: Marketing Insights Dashboard Data
print("\n💼 Generating marketing insights...")

# Load original reviews
processed_file_path = "src/data/processed/processed_reviews.json"
with open(processed_file_path, 'r', encoding='utf-8') as f:
    processed_data = json.load(f)
original_reviews = processed_data['original_reviews']

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

# Step 5.7: Save Marketing Insights
insights_path = os.path.join(viz_dir, "marketing_insights.json")
with open(insights_path, 'w', encoding='utf-8') as f:
    json.dump(marketing_insights, f, indent=2, ensure_ascii=False)
print(f"✅ Marketing insights saved to: {insights_path}")

# Step 5.8: Create Marketing Report
print("\n📋 Generating marketing report...")

report_content = f"""
# CUSTOMER REVIEW TOPIC ANALYSIS - MARKETING REPORT

## Executive Summary
- **Total Reviews Analyzed**: {len(original_reviews)}
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

report_path = os.path.join(viz_dir, "marketing_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"✅ Marketing report saved to: {report_path}")

# Step 5.9: Create New Review Classifier Function
print("\n🔮 Creating new review classifier...")

def classify_new_review(review_text):
    """
    Classify a new review using the trained LDA model
    Returns the dominant topic and business insights
    """
    # Import preprocessing function (you might need to adjust the import)
    import re
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    
    # Preprocess the new review (same as training data)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Clean text
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

# Test the classifier with a new example
test_review = "The battery dies too quickly and customer support was not helpful at all"
classification_result = classify_new_review(test_review)

print(f"\n🧪 Testing classifier with new review:")
print(f"Review: '{test_review}'")
print(f"Classification: {classification_result}")

# Save classifier function
classifier_code = '''
def classify_new_review(review_text, lda_model, dictionary, marketing_insights):
    """
    Classify a new review using the trained LDA model
    Returns the dominant topic and business insights
    """
    import re
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    
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
            'message': 'Unable to classify'
        }

# Usage:
# result = classify_new_review("New review text", lda_model, dictionary, marketing_insights)
'''

classifier_path = os.path.join(viz_dir, "review_classifier.py")
with open(classifier_path, 'w', encoding='utf-8') as f:
    f.write(classifier_code)
print(f"✅ Review classifier saved to: {classifier_path}")

print("\n" + "="*60)
print("🎉 CONGRATULATIONS! LDA PROJECT COMPLETE!")
print("="*60)
print("✅ Topic modeling pipeline built from scratch")
print("✅ Customer review topics discovered")
print("✅ Beautiful visualizations created")
print("✅ Marketing insights generated")
print("✅ New review classifier ready for use")
print("✅ Complete marketing report generated")

print(f"\n📂 All outputs saved to:")
print(f"  📊 Visualizations: {viz_dir}/")
print(f"  📋 Marketing Report: {report_path}")
print(f"  🔮 Review Classifier: {classifier_path}")
print(f"  💼 Business Insights: {insights_path}")

print(f"\n🚀 NEXT STEPS FOR MARKETING:")
print("1. Review the marketing_report.md for actionable insights")
print("2. Use the classifier to automatically categorize new reviews")
print("3. Set up monitoring for topic trends over time")
print("4. Create targeted campaigns based on discovered topics")
print("5. Integrate with your CRM/marketing automation tools")

print("\n🎯 YOU'VE SUCCESSFULLY BUILT AN LDA-POWERED MARKETING TOOL!")