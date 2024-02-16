import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stop words data (you need to do this once)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load training data
train_data_path = 'train_data.csv'  # Replace with the actual path to your training data
train_df = pd.read_csv(train_data_path)

# Data Cleaning
# Remove Duplicates
train_df = train_df.drop_duplicates()

# Handling Missing Values
train_df = train_df.dropna()

# Text Cleaning
stop_words = set(stopwords.words('english'))
train_df['description'] = train_df['description'].str.lower()
train_df['description'] = train_df['description'].str.replace('[^\w\s]', '')  # Remove punctuation
train_df['description'] = train_df['description'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))  # Remove stop words

# Remove Irrelevant Columns
train_df = train_df[['id', 'title', 'genre', 'description']]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_df['description'], train_df['genre'], test_size=0.2, random_state=42
)

# Preprocess the data (TF-IDF vectorization)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train a Logistic Regression classifier with optimized hyperparameters
logistic_regression_classifier = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
logistic_regression_classifier.fit(X_train_tfidf, y_train)

# Predict on the validation set
val_predictions = logistic_regression_classifier.predict(X_val_tfidf)

# Evaluate the performance on the validation set
print("Validation Classification Report:\n", classification_report(y_val, val_predictions, zero_division=1))
print("Validation Accuracy: ", accuracy_score(y_val, val_predictions))

# Load the test data for prediction
test_data_path = 'test_data.csv'  # Replace with the actual path to your test data
test_df = pd.read_csv(test_data_path)

# Data Cleaning for Test Data
# Apply the same text cleaning steps as for the training data
test_df['description'] = test_df['description'].str.lower()
test_df['description'] = test_df['description'].str.replace('[^\w\s]', '')
test_df['description'] = test_df['description'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

X_test_tfidf = vectorizer.transform(test_df['description'])

# Make predictions on the test data
test_predictions = logistic_regression_classifier.predict(X_test_tfidf)

# Load the test data solution for validation
test_solution_path = 'test_data_solution.csv'  # Replace with the actual path to your test data solution
test_solution_df = pd.read_csv(test_solution_path)
y_true = test_solution_df['genre']

# Evaluate the performance on the test set
print("\nTest Classification Report:\n", classification_report(y_true, test_predictions, zero_division=1))
print("Test Accuracy: ", accuracy_score(y_true, test_predictions))

# Save predictions to a file (e.g., 'test_predictions_logistic_regression.csv')
test_predictions_df = pd.DataFrame({'GENRE_PREDICTED': test_predictions})
test_predictions_df.to_csv('test_predictions_logistic_regression.csv', index=False)
