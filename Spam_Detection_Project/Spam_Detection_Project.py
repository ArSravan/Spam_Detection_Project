
import pandas as pd

# Load the spam dataset
data = pd.read_csv(r'C:\Users\ADMIN\OneDrive\Desktop\spam.csv', encoding='ISO-8859-1')

# Display basic information about the dataset
print(data.info())
print(data.head())

# Remove unwanted columns
data = data[['v1', 'v2']]
data.columns = ['Label', 'Message'] # Renaming columns

# Encode labels: 'spams' --> 1 and 'ham'--> 0
data['Label'] = data['Label'].map({'spam':1, 'ham':0})

# Check for missing values
print('Missing values:\n', data.isnull().sum())

# View the first few rows to verify changes
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer with a limit on maximum features
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')

# Transform the Messsage column into a TF-IDF matrix
x = vectorizer.fit_transform(data['Message'])

# Extract the target labels
y = data['Label']

from sklearn.model_selection import train_test_split

# Split the Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Verify the shapes
print('Training data shape:', x_train.shape)
print('Training data shape:', x_test.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the model
model = MultinomialNB()

# Train the model on the Training data
model.fit(x_train, y_train)

# Predict on the test data
y_pred = model.predict(x_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', conf_matrix)

import pandas as pd

# Load the spam dataset
data = pd.read_csv(r'C:\Users\ADMIN\OneDrive\Desktop\spam.csv', encoding='ISO-8859-1')

# Display basic information about the dataset
print(data.info())
print(data.head())

# Remove unwanted columns
data = data[['v1', 'v2']]
data.columns = ['Label', 'Message'] # Renaming columns

# Encode labels: 'spams' --> 1 and 'ham'--> 0
data['Label'] = data['Label'].map({'spam':1, 'ham':0})

# Check for missing values
print('Missing values:\n', data.isnull().sum())

# View the first few rows to verify changes
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer with a limit on maximum features
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')

# Transform the Messsage column into a TF-IDF matrix
x = vectorizer.fit_transform(data['Message'])

# Extract the target labels
y = data['Label']

from sklearn.model_selection import train_test_split

# Split the Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Verify the shapes
print('Training data shape:', x_train.shape)
print('Training data shape:', x_test.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the model
model = MultinomialNB()

# Train the model on the Training data
model.fit(x_train, y_train)

# Predict on the test data
y_pred = model.predict(x_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', report)