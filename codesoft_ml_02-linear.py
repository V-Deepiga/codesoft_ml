import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the dataset
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")

# Drop non-numeric columns or columns with non-unique values
train_data = train_data.drop(columns=['trans_num', 'trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state', 'job', 'dob'])
test_data = test_data.drop(columns=['trans_num', 'trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state', 'job', 'dob'])

# Convert categorical variables to numeric using Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['category', 'gender']
for column in categorical_columns:
    train_data[column] = label_encoder.fit_transform(train_data[column])
    test_data[column] = label_encoder.transform(test_data[column])

# Separate features and target variable
y_train = train_data['is_fraud']
X_train = train_data.drop(columns=['is_fraud'])
y_test = test_data['is_fraud']
X_test = test_data.drop(columns=['is_fraud'])

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

# Adjusting threshold to improve precision and recall
threshold = 0.5  # Adjust this threshold as needed
y_pred_test_adjusted = (y_pred_proba_test > threshold).astype(int)

# Visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test_adjusted)
plt.figure(figsize=(10, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.yticks([0, 1], ['Not Fraud', 'Fraud'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='red')
plt.title('Confusion Matrix')
plt.show()

# Plotting distribution of fraud and non-fraud transactions
plt.figure(figsize=(8, 6))
fraud_counts = [sum(y_test == 0), sum(y_test == 1)]
labels = ['Not Fraud', 'Fraud']
plt.bar(labels, fraud_counts, color=['blue', 'red'])
plt.xlabel('Transaction Type')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Fraud and Non-Fraud Transactions')
plt.show()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_test_adjusted)
print("Accuracy:", accuracy)

tp, fp, fn, tn = conf_matrix.ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
