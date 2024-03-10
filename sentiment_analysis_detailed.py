"""This program performs sentiment analysis and compares 
similarity on a dataset of Amazon product reviews using spaCy and TextBlob.
This version has more detailed comments explaining the code."""

import pkg_resources
import spacy
import pandas as pd

# Import SpacyTextBlob class & TextBlob class from the spacytextblob & textblob modules.
# These classes extends spaCy's capabilities by adding sentiment analysis features.

# Though SpacyTextBlob & TextBlob are't used directly in the code,
# functionality is being used from the spacytextblob & textblob modules.
# A functionality for example is adding the SpacyTextBlob component
# to the pipeline through a SpacyTextBlob extension.

# So if the import statements for SpacyTextBlob and TextBlob are removed
# on the basis of them not directly being used in the code
# or being accessed as shown in the linter error, then
# spaCy won't recognise the extension being used, causing an AttributeError.

from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob

print()
# Check that SpacyTextBlob & TextBlob are correctly installed into Python environment.
installed_packages = [pkg.key for pkg in list(pkg_resources.working_set)]
if 'textblob' and 'spacytextblob' in installed_packages:
    print("TextBlob and SpacyTextBlob are installed.\n")
else:
    print("TextBlob and SpacyTextBlob are not installed.\n")

# 1.
# Load small English language spaCy model, used for sentiment analysis.
# Load medium English language spaCy model, used for similarity analysis.
nlp_sm = spacy.load('en_core_web_sm')
nlp_md = spacy.load('en_core_web_md')

# Add SpacyTextBlob as a pipeline component with a string name.
# This pipeline component relies on the SpacyTextBlob extension.
nlp_sm.add_pipe('spacytextblob', last=True)

# 'last=True:', this argument specifies that the extension should be added
# at the end of the pipeline. The order in which components are added to the
# pipeline matters, and by setting last=True, it ensures that the spacytextblob
# extension is the last component in the pipeline.

# Adding the spacytextblob extension to the pipeline allows spaCy to utilise
# the sentiment analysis features provided by the extension when processing
# text through the pipeline. It means that after tokenization, part-of-speech tagging,
# and other processing steps, the text will undergo sentiment analysis
# using the spacytextblob extension.

# Dropbox link to the shared 'amazon_product_reviews.csv' file
dropbox_link = "https://www.dropbox.com/scl/fi/asv0fe91o7k1optr83aiv/amazon_product_reviews.csv?rlkey=webp8acp9j4z4hhjlv5rfuiqh&dl=0"
raw_dropbox_link = dropbox_link.replace("www.dropbox.com", "dl.dropboxusercontent.com")

# Pandas DataFrame created - Load dataset & read the CSV file into a pandas DataFrame.
df = pd.read_csv(raw_dropbox_link)

# Display information about the dataset to check for 'reviews.text' column
print()
df.info()

# 2.1 Select the 'reviews.text' column and print to view and confim correct column.
reviews_data = df['reviews.text']
print()
print(reviews_data)

# 2.2 Clean & preprocess data to prepare it for processing.

# Drop missing values in the 'reviews.text' column for an updated new DataFrame named 'clean_data'.
# Make it explicitly an independant copy using .copy so it doesn't affect the original DataFrame.
# Add deep=True to ensure a deep copy is made.
# Now modifications to clean_data won't affect the original DataFrame.

clean_data = df.dropna(subset=['reviews.text']).copy(deep=True)

# Display the data type of the 'reviews.text' column, to check it has the correct data type.
# This is so we can distinguish whether to ignore the warning sign given by Python
# in regards to mixed data types in other columns of the original 'df' DataFrame.
# As we don't need the other columns and the column we are working with has the correct
# data type as shown through the output of this print function, we can ignore the warning message.
print()
print(f"Data type of 'reviews.text' column: {df['reviews.text'].dtype}")
print()

# .lower() method standardises text and tackles case sensitivity
# .strip() method removes the whitespace before and after the token.
# .strip() may not be needed and is redundant because ' '.join() puts back the spaces.
# However, it may be used for purposes of this task.

# str() function doesn't need to be used because tokens obtained from the
# token.text attribute are inherently strings.
# Also the similarity method assumes that the content is text (i.e. strings).
# If there are integers within the text, spaCy will treat them
# as part of the string rather than as numeric values.
# str() is used for purposes of this task and to ensure explicit conversion,
# but is redundant in this case.

# These methods & function ensure text data is processed into a uniform and clean version.
def preprocess_text(text):
    """Function pre-processes text data (remove stopwords and perform basic cleaning)"""
    tokenized_text = nlp_sm(text)
    # List comprehension below iterates over each token in the spaCy Doc object (tokenized_text).
    # For each token, if the token isn't a stop word (not token.is_stop)
    # the token's text (token.text) is added to the list 'cleaned_tokens_without_stopwords'.

    # A stop word is a common word that is often filtered out in natural language processing
    # because it doesn't carry much meaning (e.g., "the," "and," "is").
    # You can't use .append here because list comprehensions are used for creating new lists,
    # not modifying existing ones.
    # The append method is for modifying exisiting lists & doesn't return a value.
    cleaned_tokens_without_stopwords = [str(token.text).lower().strip() for token in tokenized_text if not token.is_stop]
    return ' '.join(cleaned_tokens_without_stopwords)
    # Another way of returning text without stop words:
    # def remove_stopwords(text):
    #   tokenized_text = nlp(text)
    #   tokens_without_stopwords = []
    #   for token in tokenized_text:
    #       if not token.is_stop:
    #           tokens_without_stopwords.append(token.text)
    #   return ' '.join(tokens_without_stopwords)

# 3.
def analyse_sentiment(product_review):
    """Function for sentiment analysis using spaCy with TextBlob extension"""
    doc = nlp_sm(product_review)
    # Using the polarity attribute from TextBlob
    polarity = doc._.blob.polarity
    # Using the sentiment attribute from TextBlob
    sentiment = doc._.blob.sentiment

    return polarity, sentiment

def compare_similarity(review1, review2):
    """Function to compare the similarity of two product reviews"""
    doc1 = nlp_md(review1)
    doc2 = nlp_md(review2)
    similarity_score = doc1.similarity(doc2)
    return similarity_score

# Compare the similarity of two reviews from the 'reviews_text' column.
# Use indexing to select rows of choice from 'reviews_text' column.
my_review_of_choice_1 = df['reviews.text'][0]
my_review_of_choice_2 = df['reviews.text'][1]

# Call on compare_similarity function.
similarity_value = compare_similarity(my_review_of_choice_1, my_review_of_choice_2)
print(f"Review 1: {my_review_of_choice_1}")
print(f"Review 2: {my_review_of_choice_2}")
print(f"\nSimilarity Score of Two Reviews: {similarity_value}\n")

# 4. # Test sentiment analysis on a sample product reviews
sample_reviews = [
    df['reviews.text'][0],
    df['reviews.text'][1]
]

# Iterate through list of sample reviews to perform sentiment analysis.
for review in sample_reviews:
    polar, senti = analyse_sentiment(review)
    review_cleaned = preprocess_text(review)
    sentiment_result = analyse_sentiment(review_cleaned)
    print(f"Review: {review}")
    print(f"Sample Review Sentiment: {sentiment_result}")
    print(f"Polarity: {polar}, Sentiment: {senti}")
    print("\n")

# Apply the preprocess_text function to the 'reviews.text' column in the clean_data DataFrame.
# This new version of the clean_data DataFrame without stopwords & with basic cleaning, is stored in
# a new version of the clean_data DataFrame but with a new column labelled 'reviews_cleaned'.
clean_data.loc[:, 'reviews_cleaned'] = clean_data['reviews.text'].head(3).apply(preprocess_text)

# .loc[row_indexer, col_indexer], .loc helps avoid errors and explicitly indicates to pandas
# that the original DataFrame is being modified, not a copy.
# Using .loc ensures clarity and avoids potential issues.

# .loc is a label based method used for indexing in pandas,
# it allows you to access a group of rows & columns by label.
# .loc with a colon (:) explicitly indicates that you want to
# include all rows for the 'reviews_cleaned' column.

# Apply the analyse_sentiment function to the 'reviews_cleaned' column
clean_data.loc[:, 'sentiment'] = clean_data['reviews_cleaned'].head(3).apply(analyse_sentiment)

# Make sure .head(3) is added to each newly created clean_data column i.e. clean_data.loc,
# Otherwise due to a large dataset the processing will take so long it won't reach the print
# statement below. So it is sensible to only use a subset of data with the first 3 rows.
# optimise the processing time by applying the operations only to a subset of the data

print(clean_data[['reviews_cleaned', 'sentiment']].head(3))