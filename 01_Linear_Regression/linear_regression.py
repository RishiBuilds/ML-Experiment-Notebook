import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Boston.csv")

X = df[['rm']]  # Input (Number of rooms)
y = df['medv']  # Output (House price)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure()
plt.scatter(X, y)  # Actual data points
plt.plot(X, y_pred)  # Regression line
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("House Price (MEDV)")
plt.title("Linear Regression: RM vs MEDV")
plt.show()

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
