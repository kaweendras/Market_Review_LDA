# Market Review LDA Analysis

## 🎯 Overview

This project performs Latent Dirichlet Allocation (LDA) topic modeling on customer review data to extract meaningful insights for marketing strategies. The system processes raw text reviews, identifies hidden topics, and generates comprehensive visualizations and marketing recommendations.

## ✨ Key Features

- **Automated Text Preprocessing**: Intelligent cleaning, tokenization, and lemmatization of review text
- **Dynamic Topic Discovery**: LDA modeling to uncover hidden themes in customer feedback
- **Rich Visualizations**: Interactive charts and comprehensive analysis reports
- **Marketing Insights**: Actionable recommendations based on topic analysis
- **Modular Architecture**: Clean, maintainable code structure with reusable components

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Complete Analysis

```bash
python main.py
```

The interactive pipeline will guide you through:
1. Text preprocessing and data cleaning
2. Dictionary and corpus creation
3. LDA model training with optimal parameters
4. Visualization generation and insight extraction

## 📊 Sample Outputs

### Topic Distribution Analysis

The system generates comprehensive topic distribution visualizations showing how topics are distributed across your review dataset:

![Topic Distribution](https://github.com/kaweendras/Market_Review_LDA/blob/main/sample_results/topic_distribution.png?raw=true)

*Interactive bar chart showing the prevalence of each discovered topic*

### Topic Document Heatmap

Interactive heatmap visualization showing the relationship between documents and topics, with color intensity indicating topic strength:

![Topic Document Heatmap](https://github.com/kaweendras/Market_Review_LDA/blob/main/sample_results/topic_document_heatmap.png?raw=true)

*Interactive heatmap showing how topics are distributed across documents and their relative strength*


### Word Cloud Visualizations

Beautiful word clouds for each discovered topic, highlighting the most important terms:

![Word Clouds](https://github.com/kaweendras/Market_Review_LDA/blob/main/sample_results/topic_wordclouds.png?raw=true)

*Word clouds for each topic showing key terms and their relative importance*

### Marketing Insights Report

Comprehensive analysis document with actionable marketing recommendations:

![Marketing Report](https://github.com/kaweendras/Market_Review_LDA/blob/main/sample_results/marketing_report.md)

*Detailed report with topic analysis and strategic recommendations*

## 📈 What You Get

### 1. Topic Analysis
- **Automatic Topic Discovery**: Identifies 5-10 key themes in your reviews
- **Topic Coherence Scoring**: Ensures topics are meaningful and interpretable
- **Word-Topic Associations**: Shows which words are most important for each topic

### 2. Visual Analytics
- **Interactive Topic Browser**: Explore topics and their relationships
- **Distribution Charts**: See how topics are spread across your data
- **Word Importance Visualizations**: Understand what drives each topic

### 3. Marketing Intelligence
- **Customer Sentiment Themes**: Understand what customers care about most
- **Product Feature Analysis**: Identify which features generate discussion
- **Competitive Insights**: Discover gaps and opportunities in the market

### 4. Export Formats
- **HTML Reports**: Interactive web-based analysis reports
- **PNG/SVG Charts**: High-quality charts for presentations
- **CSV Data**: Raw topic assignments and probabilities for further analysis
- **Word Documents**: Professional reports ready for stakeholders

## 📁 Project Structure

```
Market_Review_LDA/
├── main.py                 # Interactive pipeline runner
├── train.py               # LDA model training
├── visualize.py          # Visualization generation
├── requirements.txt      # Dependencies
└── src/                  # Source modules
    ├── analysis/         # Text preprocessing
    ├── data/            # Data processing and storage
    ├── models/          # Trained models and artifacts
    ├── training/        # LDA training modules
    └── visualizations/  # Visualization components
```

## 🔧 Advanced Usage

### Custom Parameters

```python
from src.training.lda_trainer import train_lda_model

# Train with custom parameters
model_results = train_lda_model(
    corpus_path="src/models/review_corpus.mm",
    dictionary_path="src/models/review_dictionary.dict",
    num_topics=8,
    alpha='auto',
    eta='auto',
    passes=20,
    iterations=400
)
```

### Batch Processing

```python
from src.analysis.preprocessing import process_reviews_pipeline

# Process multiple datasets
for dataset in datasets:
    results = process_reviews_pipeline(
        csv_path=dataset,
        output_dir="src/data/processed",
        verbose=True
    )
```

## 📊 Performance Metrics

The system automatically calculates and reports:

- **Topic Coherence**: Measures how semantically similar topic words are
- **Perplexity**: Model's predictive performance on held-out data
- **Word Coverage**: Percentage of vocabulary captured by topics
- **Topic Diversity**: How distinct topics are from each other

## 🎯 Use Cases

### Marketing Teams
- **Campaign Planning**: Identify themes to focus marketing messages
- **Content Strategy**: Understand what topics resonate with customers
- **Competitive Analysis**: Discover market gaps and opportunities

### Product Teams
- **Feature Prioritization**: See which features customers discuss most
- **Quality Insights**: Identify common issues and pain points
- **Innovation Ideas**: Discover unmet needs and desires

### Customer Success
- **Support Optimization**: Understand common customer concerns
- **Training Focus**: Identify areas where customers need more help
- **Satisfaction Drivers**: Discover what makes customers happy

## 🔍 Technical Details

### Preprocessing Pipeline
- Removes HTML tags, special characters, and noise
- Handles negations and contractions appropriately
- Applies intelligent stemming and lemmatization
- Filters out stopwords and very rare/common terms

### LDA Implementation
- Uses Gensim's optimized LDA implementation
- Automatic hyperparameter tuning
- Supports both online and batch learning
- Includes model validation and coherence scoring

### Visualization Engine
- Interactive HTML visualizations using pyLDAvis
- Static charts using matplotlib and seaborn
- Word clouds with intelligent color schemes
- Responsive layouts for different screen sizes

## 📚 Documentation

Each module includes comprehensive docstrings and examples:

```python
def process_reviews_pipeline(csv_path, output_dir, top_words=10, verbose=True):
    """
    Complete pipeline for processing customer reviews.
    
    Args:
        csv_path (str): Path to CSV file with reviews
        output_dir (str): Directory to save processed data
        top_words (int): Number of top words to analyze
        verbose (bool): Whether to print progress information
    
    Returns:
        dict: Results including file paths and statistics
    """
```

---

## 👨‍💻 Author

**kaweendras**

---
