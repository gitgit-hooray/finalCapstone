"""This program performs sentiment analysis and compares 
similarity on a dataset of Amazon product reviews using spaCy and TextBlob.
Please see the file 'sentiment_analysis_detailed.py' 
for the same code but with more detailed comments."""

# Import necessary modules and classes.
import pkg_resources
import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob

# Check installation of required packages
installed_packages = [pkg.key for pkg in list(pkg_resources.working_set)]
print()
if 'textblob' and 'spacytextblob' in installed_packages:
    print("TextBlob and SpacyTextBlob are installed.\n")
else:
    print("TextBlob and SpacyTextBlob are not installed.\n")

# Load spaCy models for sentiment and similarity analysis
nlp_sm = spacy.load('en_core_web_sm')
nlp_md = spacy.load('en_core_web_md')

# Add SpacyTextBlob as a pipeline component
nlp_sm.add_pipe('spacytextblob', last=True)
# This allows spaCy to use sentiment analysis features after other processing steps.

# Dropbox link to the shared 'amazon_product_reviews.csv' file
dropbox_link = "https://www.dropbox.com/scl/fi/asv0fe91o7k1optr83aiv/amazon_product_reviews.csv?rlkey=webp8acp9j4z4hhjlv5rfuiqh&dl=0"
raw_dropbox_link = dropbox_link.replace("www.dropbox.com", "dl.dropboxusercontent.com")

# Create a pandas DataFrame by reading the CSV file
df = pd.read_csv(raw_dropbox_link)

# Display information about the dataset and the 'reviews.text' column
print()
df.info()
reviews_data = df['reviews.text']
print()
print(reviews_data)

# Clean and preprocess data
clean_data = df.dropna(subset=['reviews.text']).copy(deep=True)
print()
print(f"Data type of 'reviews.text' column: {df['reviews.text'].dtype}")
print()

# Text preprocessing function
def preprocess_text(text):
    """Remove stopwords and perform basic cleaning on text data"""
    tokenized_text = nlp_sm(text)
    cleaned_tokens_without_stopwords = [str(token.text).lower().strip() for token in tokenized_text if not token.is_stop]
    return ' '.join(cleaned_tokens_without_stopwords)

# Sentiment analysis function
def analyse_sentiment(product_review):
    """Perform sentiment analysis using spaCy with TextBlob extension"""
    doc = nlp_sm(product_review)
    polarity = doc._.blob.polarity
    sentiment = doc._.blob.sentiment
    return polarity, sentiment

# Similarity comparison function
def compare_similarity(review1, review2):
    """Compare the similarity of two product reviews"""
    doc1 = nlp_md(review1)
    doc2 = nlp_md(review2)
    similarity_score = doc1.similarity(doc2)
    return similarity_score

# Compare the similarity of two reviews from the 'reviews_text' column
my_review_of_choice_1 = df['reviews.text'][0]
my_review_of_choice_2 = df['reviews.text'][1]
similarity_value = compare_similarity(my_review_of_choice_1, my_review_of_choice_2)
print(f"Review 1: {my_review_of_choice_1}")
print(f"Review 2: {my_review_of_choice_2}")
print(f"\nSimilarity Score of Two Reviews: {similarity_value}\n")

# Test sentiment analysis on a sample of product reviews
sample_reviews = [df['reviews.text'][0], df['reviews.text'][1]]
for review in sample_reviews:
    polar, senti = analyse_sentiment(review)
    review_cleaned = preprocess_text(review)
    sentiment_result = analyse_sentiment(review_cleaned)
    print(f"Review: {review}")
    print(f"Sample Review Sentiment: {sentiment_result}")
    print(f"Polarity: {polar}, Sentiment: {senti}")
    print("\n")

# Apply text preprocessing and sentiment analysis to a subset of clean_data
clean_data.loc[:, 'reviews_cleaned'] = clean_data['reviews.text'].head(3).apply(preprocess_text)
clean_data.loc[:, 'sentiment'] = clean_data['reviews_cleaned'].head(3).apply(analyse_sentiment)

# Print the first 3 rows of the resulting clean_data DataFrame
print(clean_data[['reviews_cleaned', 'sentiment']].head(3))
