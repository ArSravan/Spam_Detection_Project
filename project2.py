import numpy as np
import pandas as pd

dataset = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\b5662effbdae8746f7f7d8ed70c42b2d-faf8b1a0d58e251f48a647d3881e7a960c3f0925\50_startups.csv")


x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder = "passthrough")

x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

np.set_printoptions(precision= 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

