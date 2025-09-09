# Market Review LDA Analysis - Modular Structure

## 📁 Project Structure

```
Market_Review_LDA/
├── main.py                    # Main entry point with interactive pipeline
├── requirements.txt          # Python dependencies
├── train.py                  # LDA model training script
├── visualize.py             # Visualization script
└── src/                     # Source code modules
    ├── analysis/           # Text analysis and preprocessing
    │   ├── __init__.py
    │   └── preprocessing.py    # Text preprocessing functions
    ├── data/               # Data processing modules
    │   ├── __init__.py
    │   ├── corpus_builder.py   # Dictionary and corpus creation
    │   ├── processed/          # Processed data files
    │   └── raw/               # Raw data files
    │       └── sample_reviews.csv
    ├── models/             # Saved models and artifacts
    │   ├── corpus_metadata.json
    │   ├── review_corpus.mm
    │   └── review_dictionary.dict
    ├── training/           # LDA training modules
    │   ├── __init__.py
    │   └── lda_trainer.py      # LDA model training functions
    └── visualizations/     # Visualization modules
        ├── __init__.py
        └── visualizer.py       # Visualization and insights functions
```

## 🚀 Usage

### Command Line Interface

The main.py script provides an interactive pipeline for the entire LDA analysis process:

#### Interactive Usage
```bash
# Run the complete interactive pipeline
python main.py
```

The script will guide you through 5 steps:
1. **Text Preprocessing** - Process and clean the raw review text data
2. **Dictionary and Corpus Creation** - Create dictionary and corpus from processed text
3. **LDA Model Training** - Train the LDA model on the processed corpus
4. **Visualizations & Marketing Insights** - Generate visualizations and extract insights
5. **Open Generated Files** - Open the created visualizations and reports

Each step requires user confirmation before proceeding, allowing you to:
- Control the execution flow
- Stop at any point if needed
- Review intermediate results
- Skip steps that have already been completed

### Function-based Usage

You can also use the functions directly in your own scripts:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.preprocessing import process_reviews_pipeline
from data.corpus_builder import create_dictionary_and_corpus_pipeline

# Step 1: Text Preprocessing
preprocessing_results = process_reviews_pipeline(
    csv_path="src/data/raw/sample_reviews.csv",
    output_dir="src/data/processed",
    top_words=10,
    verbose=True
)

# Step 2: Dictionary and Corpus Creation
corpus_results = create_dictionary_and_corpus_pipeline(
    processed_file_path=preprocessing_results['file_paths']['json_path'],
    models_dir="src/models",
    no_below=2,
    no_above=0.8,
    keep_n=1000,
    verbose=True
)

# Access results
dictionary = corpus_results['dictionary']
corpus = corpus_results['corpus']
```

## 📚 Module Details

### `src/analysis/preprocessing.py`
Functions for text preprocessing and analysis:
- `setup_nltk()`: Download NLTK dependencies
- `initialize_preprocessing_tools()`: Setup lemmatizer and stopwords
- `preprocess_text()`: Clean and tokenize text
- `load_reviews_from_csv()`: Load reviews from CSV
- `analyze_word_frequency()`: Analyze word frequencies
- `save_processed_data()`: Save results to files
- `check_data_quality()`: Validate processed data
- `process_reviews_pipeline()`: Complete preprocessing pipeline

### `src/data/corpus_builder.py`
Functions for creating dictionary and corpus:
- `load_processed_reviews()`: Load preprocessed data
- `create_dictionary()`: Create Gensim dictionary
- `filter_dictionary()`: Filter dictionary by frequency
- `create_corpus()`: Create bag-of-words corpus
- `calculate_corpus_statistics()`: Analyze corpus stats
- `save_dictionary_and_corpus()`: Save to files
- `validate_corpus()`: Validate for LDA readiness
- `analyze_word_frequencies()`: Analyze word distributions
- `create_dictionary_and_corpus_pipeline()`: Complete corpus creation pipeline

## 🔧 Benefits of Modular Structure

1. **Reusability**: Functions can be imported and used in other scripts
2. **Testability**: Each function can be tested independently
3. **Maintainability**: Code is organized by functionality
4. **Flexibility**: Parameters can be customized for different use cases
5. **CLI Interface**: Easy-to-use command line interface for common workflows
6. **Documentation**: Each function is well-documented with docstrings

## 📈 Data Flow

1. **Raw Data** → `src/data/raw/sample_reviews.csv`
2. **Preprocessing** → `src/data/processed/processed_reviews.json` & `.csv`
3. **Dictionary & Corpus** → `src/models/review_dictionary.dict` & `review_corpus.mm`
4. **LDA Training** → Use existing `train.py` script
5. **Visualization** → Use existing `visualize.py` script

## 🔄 Migration from Old Structure

The old monolithic scripts (`analyse.py`, `data_processing.py`) have been refactored into:
- Modular functions in appropriate directories
- A unified CLI interface in `main.py`
- Proper Python package structure with `__init__.py` files
- Better error handling and validation
- Comprehensive documentation and examples

## 🎯 Next Steps

1. Use `python main.py` to run the complete interactive pipeline
2. Follow the guided steps for preprocessing, corpus creation, and LDA training
3. Review the generated visualizations and marketing insights
4. Extend the modules with additional functionality as needed
