import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

# Load the dataset
print("Loading dataset...")
dataset = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\Fraud.csv")

# Define the feature matrix X and target vector y
print("Defining feature matrix X and target vector y...")
x = dataset.iloc[:, :-2]
y = dataset.iloc[:, -2].values

# Specify categorical and numerical columns
categorical_columns = ["type", "nameOrig", "nameDest"]
numerical_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Define the ColumnTransformer with OneHotEncoder for categorical columns and passthrough for numerical columns
print("Applying ColumnTransformer...")
ct = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), categorical_columns),
    ("num", "passthrough", numerical_columns)
])

# Transform the feature matrix X
x_transformed = ct.fit_transform(x)

# Split the data into training and testing sets
print("Splitting dataset into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.2, random_state=42)

# Scale the numerical features
print("Standardizing numerical features...")
sc = StandardScaler(with_mean=False)

# Get the indices of numerical columns in the transformed dataset
num_indices = np.arange(len(categorical_columns), len(categorical_columns) + len(numerical_columns))

# Extract numerical features and scale them
x_train_num = sc.fit_transform(x_train[:, num_indices])
x_test_num = sc.transform(x_test[:, num_indices])

# Combine scaled numerical features with the rest of the sparse matrix
x_train_combined = hstack((x_train[:, :len(categorical_columns)], x_train_num))
x_test_combined = hstack((x_test[:, :len(categorical_columns)], x_test_num))

# Train a Random Forest classifier
print("Training Random Forest classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train_combined, y_train)

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred = rf_classifier.predict(x_test_combined)

# Print the predictions
print("Predictions:", y_pred)

