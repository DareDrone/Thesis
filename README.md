# Thesis
Amazon Customer Reviews
# Loading data into Python
real_reviews = pd.read_excel('C:/Users/C.data.Final.xlsx')
ai_reviews = pd.read_excel('C:/Users/AI Generated Data.xlsx')
# Filling the missing values in text columns with 'No Text'
real_reviews['Text Heading'].fillna('No Text', inplace=True)
real_reviews['Review'].fillna('No Text', inplace=True)

# To make it so that the numeric columns don't collide, it is better to identify numeric columns
numeric_columns = real_reviews.select_dtypes(include=['number']).columns

# Fill missing values in numeric columns with the mean of each column
for col in numeric_columns:
    real_reviews[col].fillna(real_reviews[col].mean(), inplace=True)
# Loading data into Python
real_reviews = pd.read_excel('C:/Users/C.data.Final.xlsx')
ai_reviews = pd.read_excel('C:/Users/AI Generated Data.xlsx')
# Filling the missing values in text columns with 'No Text'
real_reviews['Text Heading'].fillna('No Text', inplace=True)
real_reviews['Review'].fillna('No Text', inplace=True)

# To make it so that the numeric columns don't collide, it is better to identify numeric columns
numeric_columns = real_reviews.select_dtypes(include=['number']).columns

# Fill missing values in numeric columns with the mean of each column
for col in numeric_columns:
    real_reviews[col].fillna(real_reviews[col].mean(), inplace=True)

# Preprocessing Data And EDA
#Merging the heading and long text column
real_reviews['merged_text'] = real_reviews['Text Heading'].astype(str) + " " + real_reviews['Review'].astype(str)

def clean_text(text):
    # Converting text to lowercase
    text = text.lower()
    # Removing non-alphabetic characters and splitting the text into words
    return ' '.join(re.sub(r'[^a-zA-Z]', ' ', text).split())

# Applying the function to your text column
ai_reviews['cleaned_text1'] = ai_reviews['Constructive Feedback'].apply(clean_text)
real_reviews['cleaned_text2'] = real_reviews['merged_text'].apply(clean_text)
#Finding duplicates
# Finding duplicate rows based on all columns
duplicate_rows_real = real_reviews.duplicated()
duplicate_rows_ai = ai_reviews.duplicated()

# Print the number of duplicate rows
print(f'Number of duplicate rows in real_reviews: {duplicate_rows_real.sum()}')
print(f'Number of duplicate rows in ai_reviews: {duplicate_rows_ai.sum()}')

# Finding missing values in each column
missing_values_real = real_reviews.isnull().sum()
missing_values_ai = ai_reviews.isnull().sum()

# Print the number of missing values for each column
print(f'Missing values in real_reviews:\n{missing_values_real}')
print(f'Missing values in ai_reviews:\n{missing_values_ai}')

# Checking quantiles
# Describing the data to get the quantiles
quantiles_real = real_reviews.describe()
quantiles_ai = ai_reviews.describe()

# Print the quantiles
print(f'Quantiles for real_reviews:\n{quantiles_real}')
print(f'Quantiles for ai_reviews:\n{quantiles_ai}')

duplicate_text_real = real_reviews.duplicated(subset=['cleaned_text2'])
duplicate_text_ai = ai_reviews.duplicated(subset=['cleaned_text1'])

# Print the number of duplicate text entries
print(f'Number of duplicate text entries in real_reviews: {duplicate_text_real.sum()}')
print(f'Number of duplicate text entries in ai_reviews: {duplicate_text_ai.sum()}')

# Removing duplicates in the text column
real_reviews = real_reviews[~duplicate_text_real]
ai_reviews = ai_reviews[~duplicate_text_ai]

# Resetting the index after removing duplicates
real_reviews.reset_index(drop=True, inplace=True)
ai_reviews.reset_index(drop=True, inplace=True)

total_entries_real = real_reviews.shape[0]
total_entries_ai = ai_reviews.shape[0]

print(f'Total entries in real_reviews: {total_entries_real}')
print(f'Total entries in ai_reviews: {total_entries_ai}')

# Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)  
# Adjust max_features to your preference
tfidf_real = vectorizer.fit_transform(real_reviews['cleaned_text2'])
tfidf_ai = vectorizer.fit_transform(ai_reviews['cleaned_text1'])

# Getting the feature names and printing them
feature_names = vectorizer.get_feature_names_out()
print(f'Key Features: {feature_names}')

# Sentiment Analysis
pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create a VADER sentiment intensity analyzer object
analyzer = SentimentIntensityAnalyzer()

# Define a function to calculate sentiment
def get_vader_sentiment(text):
    # Get the sentiment scores dictionary
    scores = analyzer.polarity_scores(text)
    # Return the compound score
    return scores['compound']

# Apply the function to the cleaned text columns
real_reviews['vader_sentiment_score'] = real_reviews['cleaned_text2'].apply(get_vader_sentiment)
ai_reviews['vader_sentiment_score'] = ai_reviews['cleaned_text1'].apply(get_vader_sentiment)

# Categorize sentiment into positive, neutral, and negative
def categorize_vader_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the categorization function to the VADER sentiment score columns
real_reviews['vader_sentiment_category'] = real_reviews['vader_sentiment_score'].apply(categorize_vader_sentiment)
ai_reviews['vader_sentiment_category'] = ai_reviews['vader_sentiment_score'].apply(categorize_vader_sentiment)

# Get the sentiment distribution for real reviews
real_sentiment_distribution = real_reviews['vader_sentiment_category'].value_counts()

# Print the sentiment distribution for real reviews
print("Sentiment distribution in real reviews:")
print(real_sentiment_distribution)

# Print a separator for better readability
print("\n" + "-"*50 + "\n")

# Get the sentiment distribution for AI reviews
ai_sentiment_distribution = ai_reviews['vader_sentiment_category'].value_counts()

# Print the sentiment distribution for AI reviews
print("Sentiment distribution in AI reviews:")
print(ai_sentiment_distribution)

# Analyzing sentiment distribution
sentiment_distribution_real = real_reviews['vader_sentiment_category'].value_counts(normalize=True)
sentiment_distribution_ai = ai_reviews['vader_sentiment_category'].value_counts(normalize=True)

# Calculating mean sentiment score
mean_sentiment_real = real_reviews['vader_sentiment_score'].mean()
mean_sentiment_ai = ai_reviews['vader_sentiment_score'].mean()

import matplotlib.pyplot as plt

# Visualizing sentiment distribution
sentiment_distribution_real.plot(kind='bar', title='Sentiment Distribution - Real Reviews')
plt.show()

sentiment_distribution_ai.plot(kind='bar', title='Sentiment Distribution - AI Generated Reviews')
plt.show()

# Assuming 'brand' is the column with brand information
mean_sentiment_by_brand_real = real_reviews.groupby('Products Name')['vader_sentiment_score'].mean()
mean_sentiment_by_brand_ai = ai_reviews.groupby('Product')['vader_sentiment_score'].mean()

# Visualizing sentiment by brand
mean_sentiment_by_brand_real.plot(kind='bar', title='Mean Sentiment by Brand - Real Reviews')
plt.show()

mean_sentiment_by_brand_ai.plot(kind='bar', title='Mean Sentiment by Brand - AI Generated Reviews')
plt.show()


# Visualizing sentiment by brand
# Plotting mean sentiment score by brand for real reviews
plt.figure(figsize=(10, 6))
brand_sentiment_real.sort_values().plot(kind='bar')
plt.title('Mean Sentiment Score by Brand in Real Reviews')
plt.ylabel('Mean Sentiment Score')
plt.show()

# Plotting mean sentiment score by brand for AI reviews
plt.figure(figsize=(10, 6))
brand_sentiment_ai.sort_values().plot(kind='bar')
plt.title('Mean Sentiment Score by Brand in AI Reviews')
plt.ylabel('Mean Sentiment Score')
plt.show()

# EDA OF DATASET

# Assuming real_reviews and ai_reviews are your dataframes and 'Score out of 5' and 'Star Rating' are the columns with star ratings
real_star_distribution = real_reviews['Score out of 5'].value_counts(normalize=True).sort_index()
ai_star_distribution = ai_reviews['Star Rating'].value_counts(normalize=True).sort_index()

# Colors
 # Variations of blue
colors_real = ['#cce5ff', '#99ccff', '#66b3ff', '#3399ff', '#0080ff']
 # Variations of red
colors_ai = ['#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ff0000'] 

# Plotting Real Reviews
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.pie(real_star_distribution, labels=real_star_distribution.index, autopct='%1.1f%%', startangle=140, colors=colors_real)
plt.title('Star Distribution - Real Reviews')

# Plotting AI Reviews
plt.subplot(1, 2, 2)
plt.pie(ai_star_distribution, labels=ai_star_distribution.index, autopct='%1.1f%%', startangle=140, colors=colors_ai)
plt.title('Star Distribution - AI Reviews')

plt.tight_layout()
plt.show()

# Adjusting the regular expression pattern to match the format of the data available in 
pattern = r'Reviewed in (?P<Country>.*?) on (?P<Date>\d{1,2} \w+ \d{4})'

# Use str.extract() to split the Location_Date column into separate Country and Date columns
split_data = real_reviews['Location_Date'].str.extract(pattern)

# If the extraction is successful, the split_data DataFrame will have two columns: Country and Date
# You can then concatenate these columns to your original DataFrame
real_reviews = pd.concat([real_reviews, split_data], axis=1)

# Optionally, you can drop the original Location_Date column
real_reviews.drop(columns=['Location_Date'], inplace=True)

# Now real_reviews has separate columns for Country and Date
print(real_reviews[['Country', 'Date']].head())

# Convert the 'Date' column to datetime format
real_reviews['Date'] = pd.to_datetime(real_reviews['Date'])

# Group data by Brand and Date, then calculate the mean sentiment score
time_series_analysis = real_reviews.groupby(['Brand_reviews', real_reviews['Date'].dt.to_period("M")])['sentiment_score'].mean().unstack()

# Plotting the time series analysis
time_series_analysis.transpose().plot(figsize=(10, 6))
plt.title('Sentiment Analysis Over Time by Brand')
plt.ylabel('Mean Sentiment Score')
plt.xlabel('Date')
plt.legend(title='Brand')
plt.show()

# Geographical Analysis
geo_analysis.plot(kind='bar', figsize=(10, 6))
plt.title('Sentiment Analysis by Country and Brand')
plt.ylabel('Mean Sentiment Score')
plt.xlabel('Brand')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(real_reviews['Score out of 5'], real_reviews['How Many People found it helpful'], alpha=0.5)
plt.title('Review Score vs Helpful Count')
plt.xlabel('Review Score (out of 5)')
plt.ylabel('Helpful Count')
plt.grid(True)
plt.show()

# Comparative Analysis

sentiment_comparison = pd.concat([
    real_reviews['sentiment_category'].value_counts(normalize=True).rename('Real Reviews'),
    ai_reviews['sentiment_category'].value_counts(normalize=True).rename('AI Reviews')
], axis=1)

print(sentiment_comparison)

# You can also visualize this comparison using a bar plot
sentiment_comparison.plot(kind='bar', figsize=(10, 6))
plt.title('Comparative Analysis of Sentiment Distribution')
plt.ylabel('Proportion')
plt.show()

from scipy.stats import chi2_contingency

# Chi-Square Test for Sentiment Category Distribution
contingency_table = pd.crosstab(
    real_reviews['sentiment_category'], 
    ai_reviews['sentiment_category']
)

chi2_stat, p_val, dof, ex = chi2_contingency(contingency_table)
print(f'Chi2 Stat: {chi2_stat}, P Value: {p_val}')

# Machine Learning

# Using Undersampling for the analysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Assume sentiment_category is your target variable and tf-idf features are your input features
X_real = tfidf_real  
# Assuming tfidf_real is your TF-IDF feature matrix for real reviews
y_real = real_reviews['sentiment_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

# Convert sparse matrix to dense DataFrame
X_train_df = pd.DataFrame(X_train.toarray())

# Reset index on y_train to match X_train_df
y_train_reset = y_train.reset_index(drop=True)

# Combine them back into one dataset
training_data = pd.concat([X_train_df, y_train_reset], axis=1)

# Separate majority and minority classes
majority = training_data[training_data.sentiment_category=='Positive']
minority = training_data[training_data.sentiment_category!='Positive']

# Downsample majority class
majority_downsampled = resample(majority, 
                                replace=False,    # sample without replacement
                                n_samples=len(minority),     # to match minority class
                                random_state=42) # reproducible results

# Combine minority class with downsampled majority class
downsampled = pd.concat([majority_downsampled, minority])

# Splitting the features and labels
X_train_downsampled = downsampled.drop('sentiment_category', axis=1)
y_train_downsampled = downsampled['sentiment_category']

# Decision Tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train_downsampled, y_train_downsampled)
y_pred_dt = dt.predict(X_test.toarray())  # Convert sparse matrix to dense array for prediction


# Evaluate Decision Tree model
print("Decision Tree Classifier:")
print(accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, zero_division=1))  # Added zero_division parameter


# Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_downsampled, y_train_downsampled)
y_pred_nb = nb.predict(X_test.toarray())  # Convert sparse matrix to dense array for prediction

# Evaluate Naive Bayes model
print("Naive Bayes Classifier:")
print(accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, zero_division=1))  # Added zero_division parameter

# Topic Modelling:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Assuming real_reviews['cleaned_text2'] is the text data you want to analyze
text_data = real_reviews['cleaned_text2']

# Create a count vectorizer instance
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Fit and transform the text data to get the count vector
count_vector = count_vectorizer.fit_transform(text_data)

# Now fit the LDA model
lda = LDA(n_components=5, random_state=42)  # Adjust n_components to the number of topics you want to extract
lda.fit(count_vector)

# Display topics
for idx, topic in enumerate(lda.components_):
    print(f'Topic #{idx + 1}:')
    print([count_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')

    # Data for AI-generated reviews
text_data_ai = ai_reviews['cleaned_text1']

# Transforming the AI-generated text data using the same count vectorizer
count_vector_ai = count_vectorizer.transform(text_data_ai)

# Transforming the count vector to topic distribution
topic_distribution_real = lda.transform(count_vector)
topic_distribution_ai = lda.transform(count_vector_ai)

# Converting to DataFrame for an easier analysis
topic_distribution_real_df = pd.DataFrame(topic_distribution_real, columns=[f'Topic {i+1}' for i in range(5)])
topic_distribution_ai_df = pd.DataFrame(topic_distribution_ai, columns=[f'Topic {i+1}' for i in range(5)])

# A DataFrame where each row represents a review and each column represents the proportion of each topic in that review.

sns.histplot(data=real_reviews, x='sentiment_score', kde=True, bins=30)
plt.title('Sentiment Distribution - Real Reviews')
plt.show()

sns.histplot(data=ai_reviews, x='sentiment_score', kde=True, bins=30)
plt.title('Sentiment Distribution - AI Generated Reviews')
plt.show()

# Selecting only the numeric columns (excluding the 'Dominant_Topic' column)
mean_distribution_real = topic_distribution_real_df.drop(columns='Dominant_Topic').mean()
mean_distribution_ai = topic_distribution_ai_df.drop(columns='Dominant_Topic').mean()

# Plotting the mean distribution of topics
mean_distribution_real.plot(kind='bar', title='Mean Topic Distribution - Real Reviews')
plt.show()

mean_distribution_ai.plot(kind='bar', title='Mean Topic Distribution - AI Generated Reviews')
plt.show()

