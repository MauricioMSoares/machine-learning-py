import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


data = pd.read_csv("car_purchasing_data.csv")
print(data)

x = data[["gender", "age", "salary", "debt", "worth"]]
y = data["amount"]

model = LinearRegression()
model.fit(x, y)

new_data = np.array([1, 33, 75500, 15645, 335845])
predicted_price = model.predict([new_data])
print("Predicted purchasing price: ", predicted_price)