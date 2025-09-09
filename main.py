import pandas as pd
import numpy as np
import nltk

print("✅ All libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"NLTK version: {nltk.__version__}")

# Download required NLTK data (this might take a minute the first time)
print("\nDownloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    print("✅ NLTK data downloaded successfully!")
except:
    print("❌ Issue downloading NLTK data - check your internet connection")

# Create some sample customer reviews to work with
sample_reviews = [
    "The battery life is terrible, dies within 2 hours of use.",
    "Amazing camera quality! Takes stunning photos even in low light.",
    "Customer service was unhelpful when I had issues.",
    "Great value for money. Battery lasts all day.",
    "Shipping was super fast, arrived next day.",
    "The camera features are outstanding and easy to use.",
    "Battery performance is poor, needs constant charging.",
    "Customer support team was very helpful and quick.",
    "Expensive but worth it. Great camera and battery life.",
    "Delivery was delayed but product quality is excellent."
]

print(f"\n✅ Created {len(sample_reviews)} sample reviews to work with")
print("\nSample review:")
print(f"'{sample_reviews[0]}'")

print("\n" + "="*50)
print("STEP 1 COMPLETE!")
print("="*50)
print("✅ Libraries installed and imported")
print("✅ NLTK data downloaded") 
print("✅ Sample data created")
print("\nType 'next' when you're ready for Step 2: Text Preprocessing")