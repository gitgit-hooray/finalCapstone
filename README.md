# NLP Sentiment Analysis & Similarity Comparison of Amazon Product Reviews

## Description

This project performs sentiment analysis and similarity comparison on a dataset of Amazon product reviews using spaCy and TextBlob. 
The primary goal is to analyze the sentiment of reviews and measure the similarity between different reviews, 
providing valuable insights into customer opinions and product relations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Credits](#credits)

## Installation

To use this project locally, follow these steps:

1. Open up your computer terminal and enter the following points into the command line:

2. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/gitgit-hooray/finalCapstone.git

3. Navigate to the project directory:
   ```bash
   cd "finalCapstone"

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
After installing the project, you can use it as follows:

Run the main Python scripts:

This is the original codebase with comments
```bash
python sentiment_analysis.py
```
This is an alternative version with more detailed comments explaining the code.
This is to improve understanding as a learning process and for more clarity on why certain code was used.
```bash
python sentiment_analysis_detailed.py
```

Follow the prompts to input the file path of the Amazon product reviews dataset.
The program will perform sentiment analysis and similarity comparison on the reviews, providing results and insights.

## Screenshots

### Main program code designed for sentiment analysis and similarity comparison.
<br>
1. The first part of the program sets up the necessary packages and models for sentiment analysis using spaCy and TextBlob, 
checking for package installations and configuring spaCy's pipeline for sentiment analysis.
<br>
<br>
<img width="984" alt="1" src="https://github.com/gitgit-hooray/finalCapstone/assets/151678204/82eb9526-c3fd-4b4d-8bce-faca36c8ffe9">
<br>
<br>
<br>
2. The next part of the program retrieves Amazon product reviews from a Dropbox link, creates a Pandas DataFrame for it, 
displays information about the dataset, cleans and preprocesses the data by removing null values, and defines a text 
preprocessing function to eliminate stopwords and perform basic cleaning for analysis.
<br>
<br>
<img width="976" alt="2" src="https://github.com/gitgit-hooray/finalCapstone/assets/151678204/af8ab761-ab0c-4421-ae64-2939c2b90439">
<br>
<br>
<br>
3. In this part of the code, sentiment analysis is performed using spaCy with the TextBlob extension for individual product reviews, 
and a function is implemented to compare the similarity between two reviews, with a demonstration of similarity comparison 
and sentiment analysis on a sample of product reviews from the dataset.
<br>
<br>
<img width="981" alt="3" src="https://github.com/gitgit-hooray/finalCapstone/assets/151678204/39be01c5-aa92-49d2-a09f-c8282280cb12">
<br>
<br>
<br>
4. This final part of the code, tests sentiment analysis on a sample of product reviews, displaying original and preprocessed 
versions with corresponding polarity and sentiment values. It also applies text preprocessing and sentiment analysis to a 
subset of cleaned data, showcasing the resulting DataFrame for the first three rows.
<br>
<br>
<img width="987" alt="4" src="https://github.com/gitgit-hooray/finalCapstone/assets/151678204/c49fe2dd-25ab-4c48-996e-62a8a82122cd">
<br>
<br>
<br>

### Output Results
<br>
1. This output shows two product reviews and their similarity score, indicating a relatively high similarity between the two reviews. 
The sentiment analysis is then conducted on the first review, showing a slightly negative polarity (-0.05) 
and a subjective sentiment with a subjectivity score of 0.7833.
<br>
<br>
<img width="945" alt="Sentiment Analysis" src="https://github.com/gitgit-hooray/finalCapstone/assets/151678204/1cdcd29c-fb3e-4d13-8875-2c40f779bf11">
<br>
<br>
<br>
2. Finally this output shows a cleaned version of a product review, followed by sentiment analysis results. 
The original review expresses positivity about the product, resulting in a sentiment score of 0.8 with a subjectivity of 0.825. 
This sentiment analysis is consistent with the values presented in the cleaned data DataFrame for the corresponding review.
<br>
<br>
<img width="811" alt="Sentiment Analysis 2" src="https://github.com/gitgit-hooray/finalCapstone/assets/151678204/6de9f5c2-dd0d-4c17-964e-d63deb5ea1ba">
<br>
<br>
<br>

## Credits
This project was created by Charmaine Fernandes. 

I appreciate the contributions and support from the open-source community.


