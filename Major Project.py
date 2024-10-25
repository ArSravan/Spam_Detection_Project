import numpy as np
import pandas as pd

dataset = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\Fraud.csv")

x = dataset.iloc[:, :-2]
y = dataset.iloc[:, -2].values

categorical_columns = ["type", "nameOrig", "nameDest"]
numerical_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[("cat", OneHotEncoder(), categorical_columns), ("num", "passthrough", numerical_columns)])
x = ct.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
num_indices = np.arange(len(categorical_columns), len(categorical_columns) + len(numerical_columns))

x_train = sc.fit_transform(x_train[:, num_indices])
x_test = sc.transform(x_test[:, num_indices])

from scipy.sparse import hstack
x_train_combined = hstack((x_train[:, :len(categorical_columns)], x_train))
x_test_combined = hstack((x_test[:, :len(categorical_columns)], x_test))

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators= 100, random_state=42)
rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))



















