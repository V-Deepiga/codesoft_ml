# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Data preprocessing
# Drop irrelevant columns or columns with unique identifiers (like CustomerID)
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Split data into features and target variable
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation

# Logistic Regression
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train, y_train)
lr_pred = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)
gb_pred = gb_classifier.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print("\nGradient Boosting Accuracy:", gb_accuracy)
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))

# Plot ROC curves
plt.figure(figsize=(10, 8))

# Logistic Regression ROC curve
lr_probs = lr_classifier.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, lr_probs)
auc_lr = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')

# Random Forest ROC curve
rf_probs = rf_classifier.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, rf_probs)
auc_rf = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_rf:.2f})')

# Gradient Boosting ROC curve
gb_probs = gb_classifier.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, gb_probs)
auc_gb = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Gradient Boosting (AUC = {auc_gb:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Input customer data for prediction
print("\nEnter customer data for prediction:")
credit_score = float(input("Credit Score: "))
age = float(input("Age: "))
tenure = float(input("Tenure: "))
balance = float(input("Balance: "))
num_of_products = float(input("Number of Products: "))
has_cr_card = float(input("Has Credit Card (0 for No, 1 for Yes): "))
is_active_member = float(input("Is Active Member (0 for No, 1 for Yes): "))
estimated_salary = float(input("Estimated Salary: "))
geography = input("Geography (France, Germany, Spain): ")
gender = input("Gender (Male, Female): ")

# Transform input data into the format expected by the model
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography_Germany': [1 if geography == 'Germany' else 0],
    'Geography_Spain': [1 if geography == 'Spain' else 0],
    'Gender_Male': [1 if gender == 'Male' else 0]
})

# Make prediction
prediction = gb_classifier.predict(input_data)[0]
if prediction == 1:
    print("\nThe customer is likely to churn.")
else:
    print("\nThe customer is not likely to churn.")
