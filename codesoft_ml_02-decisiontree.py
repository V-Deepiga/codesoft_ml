import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

# Train the model
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['Not Fraud', 'Fraud'])
plt.show()

# Predictions on the test set
y_pred_test = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
