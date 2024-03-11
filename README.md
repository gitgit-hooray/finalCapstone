# NLP Sentiment Analysis & Similarity Comparison of Amazon Product Reviews

## Description

This project involves performing sentiment analysis and similarity comparison on a dataset of Amazon product reviews using spaCy and TextBlob. The primary objective is to analyze sentiment, measure review similarity, and gain valuable insights into customer opinions and product relations. The dataset provides raw structured information, including product names, brands, categories, keys (product codes or unique identifiers), review dates, and texts. By isolating specific fields and cleaning the data, it becomes suitable for various applications like sentiment analysis, product recommendation, and more.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
  - [Main Program](#main-program)
  - [Output Results](#output-results)
- [Preprocessing Steps](#preprocessing-steps)
- [Evaluation of Results](#evaluation-of-results)
- [Insights into the Model's Strengths and Limitations](#insights-into-the-models-strengths-and-limitations)
  - [Model Strengths](#model-strengths)
  - [Model Limitations](#model-limitations)
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

Run the main Python scripts in terminal
<br>
<br>
This script is the original codebase with comments:

```bash
python sentiment_analysis.py
```
<br>
This script is an alternative version with more detailed comments explaining the code:
    
This is to improve an understanding of NLP as a learning process and for more clarity on why certain code was used.

```bash
python sentiment_analysis_detailed.py
```
<br>
Follow the prompts to input the file path of the Amazon product reviews dataset.
The program will perform sentiment analysis and similarity comparison on the reviews, providing results and insights.

## Screenshots

### Main Program
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

## Preprocessing Steps

1. **Remove Irrelevant Information:**
   - Preprocessing steps remove irrelevant information (stop words) and standardize the text, making it more suitable for sentiment analysis.

2. **Create 'reviews_cleaned' Column:**
   - The cleaned text is stored in a new column called 'reviews_cleaned' in the `clean_data` DataFrame.

3. **Tokenization Using spaCy:**
   - First, the text is tokenized using spaCy's small English model (`nlp_sm`).

4. **Check for Stop Words:**
   - Each token is checked to determine whether it is a stop word (e.g., "the," "and," etc.).

5. **Clean and Lowercase Tokens:**
   - If the token isn’t a stop word, its text is converted to lowercase and stripped of leading and trailing whitespaces, creating a list of cleaned tokens without stop words.

6. **Join Cleaned Tokens:**
   - The cleaned tokens are then joined back into a single string with spaces in between, simplifying the text for further analysis.

7. **Apply 'preprocess_text' Function:**
   - Finally, the ‘preprocess_text’ function is applied to the ‘reviews.text’ column of the `clean_data` DataFrame. Only the first three rows of the column are processed using `head(3)` to optimize processing time.

## Evaluation of Results

### Sentiment Analysis and Similarity Comparison

The model conducts sentiment analysis and similarity comparison of Amazon product reviews using spaCy and TextBlob.

### Library Installation Check

- Libraries (`spacy`, `textblob`, and `spacytextblob`) are checked if they are correctly installed. 
- If not, the code prints a message indicating that these libraries need to be installed. This is good practice to ensure dependencies are met before proceeding.

### Model Initialization

- Two models from spaCy were loaded, both the small and medium models for sentiment and similarity analysis, respectively.
- Different models were used depending on the appropriateness for different purposes. For example, sentiment analysis can be done with a smaller model, whereas similarity analysis produces more accurate results with a medium model.
- The `SpacyTextBlob` extension is added to the spaCy pipeline to enhance sentiment analysis capabilities. This initialization is needed to tap into spaCy's NLP functionalities.

### Sample Analysis and Preprocessing

- The code successfully applies sentiment analysis and similarity comparison to a sample of reviews.
- The preprocessing steps, including stop word removal, lowercase conversion, and text cleaning, contribute to more meaningful sentiment analysis results.
- The code currently focuses on a small subset of data (three rows) for demonstration and optimizing execution process time, especially for large datasets.

### Sentiment Analysis Challenges

- For row [0] of the ‘reviews.text’ column, the sentiment analysis result is ‘(-0.05000000000000001, Sentiment(polarity=-0.05000000000000001, subjectivity=0.7833333333333333)’, indicating a negative sentiment.
- However, the analysis fails to capture nuanced sentiment, as the removal of stop words, including 'not,' leads to an inaccurate assessment.
- 'Not disappointed' is transformed into 'disappointed,' impacting the accuracy of the sentiment analysis. This suggests that though preprocessing the data and removing stop words may be more efficient for data processing and analysis, in this case it can affect the accuracy of the sentiment analysis.

### Result Interpretation and Visualization

- To enhance result interpretation, introducing data visualizations, such as sentiment distribution plots or similarity score scatter plots, could provide a clearer understanding of the sentiment trends and relationships between reviews.

## Insights into the Model's Strengths and Limitations

### Model's Strengths

1. **Combined Strengths of spaCy and TextBlob:**
   - The code leverages both spaCy and TextBlob, combining their strengths. SpaCy handles tokenization, part-of-speech tagging, and other NLP tasks, while TextBlob provides the interface for sentiment analysis.

2. **Effective Preprocessing Steps:**
   - Preprocessing steps, including stop word removal and basic text cleaning, contribute to more meaningful sentiment analysis results. These steps help eliminate unnecessary information and focus on the essential content of the reviews.

3. **SpacyTextBlob Extension:**
   - The addition of the SpacyTextBlob extension to the spaCy pipeline enhances sentiment analysis capabilities. It allows easy integration of TextBlob's sentiment analysis features within the spaCy processing pipeline.

4. **Efficient Processing:**
   - The model is efficient by processing only the first 3 rows of data for sentiment analysis and similarity comparison. This facilitates quicker testing and debugging, especially when working with large datasets.

5. **Informative Comments:**
   - The code includes informative comments that explain the rationale behind each step. This makes the code easy to understand for both the original developer and potential collaborators.

### Model's Limitations

1. **Subset Processing and Dataset Representation:**
   - The code processes only a small subset of the dataset (3 rows) to optimize processing time, but it may not represent diversity in the entire dataset. For a more comprehensive analysis, the code needs modification to handle the entire dataset efficiently.

2. **Dependency on Stop Word Removal:**
   - The effectiveness of the sentiment analysis heavily relies on stop word removal. While removing stop words can enhance analysis, it may also lead to the loss of context, especially in situations where stop words carry valuable information. For example, as mentioned above, the removal of 'not' causes sentiment to change from positive to negative.

3. **Dependency on Input Text Quality:**
   - The model's performance is highly dependent on the quality of the input text. If the reviews contain misspellings, slang, or non-standard language, the sentiment analysis and similarity comparison may be affected.

4. **Language Assumption:**
   - The code assumes that the reviews are in English, as it loads English language models. If the dataset includes reviews in different languages, additional language-specific models and preprocessing steps would be required.

5. **Lack of Detailed Evaluation Metrics:**
   - The model lacks detailed evaluation metrics such as precision, recall, or F1 score. Incorporating these metrics would enhance the assessment of sentiment analysis accuracy.

Understanding these strengths and limitations is crucial for refining the model and making informed decisions about its application, scalability, and potential areas of improvement.
<br>
<br>
## Credits
This project was created by Charmaine Fernandes. 

I appreciate the contributions and support from the open-source community.


