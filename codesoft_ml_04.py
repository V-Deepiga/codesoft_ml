import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("spam.csv", encoding='latin1')

# Preprocessing: Drop unnecessary columns
data = data.drop(columns=data.columns[data.columns.str.contains('^Unnamed')])

# Handling missing values (if any)
data.dropna(inplace=True)

# Preprocessing: Rename columns
data.columns = ['label', 'message']

# Preprocessing: Convert labels to binary (0 for ham, 1 for spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Feature extraction: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Models training
logreg_model = LogisticRegression()
nb_model = MultinomialNB()
svm_model = SVC()

logreg_model.fit(X_train_tfidf, y_train)
nb_model.fit(X_train_tfidf, y_train)
svm_model.fit(X_train_tfidf, y_train)

# Models evaluation
models = {'Logistic Regression': logreg_model, 'Naive Bayes': nb_model, 'SVM': svm_model}
accuracies = {}
for name, model in models.items():
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    
    # Print accuracy
    print(f"{name} Accuracy:", accuracy)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print()

# Plotting individual model accuracies
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'orange', 'green'])
plt.title('Accuracy Comparison of Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.9 * min(accuracies.values()), 1.1 * max(accuracies.values()))
plt.show()

# Predicting on new data using the best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

new_messages = ["Get a free iPhone now!", "Hey, how are you doing?"]
new_messages_tfidf = tfidf_vectorizer.transform(new_messages)
predictions = best_model.predict(new_messages_tfidf)
for msg, label in zip(new_messages, predictions):
    print(f"{best_model_name} - Message: {msg} \nPredicted Label: {'spam' if label == 1 else 'ham'}")
