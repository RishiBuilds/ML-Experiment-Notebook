import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Iris.csv")

# Input and output
X = df[['PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

y = y.apply(lambda x: 1 if x == 'Iris-setosa' else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Check Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Graph:
plt.figure()
plt.scatter(X[y == 1]['PetalLengthCm'], X[y == 1]['PetalWidthCm'], label="Iris-setosa")
plt.scatter(X[y == 0]['PetalLengthCm'], X[y == 0]['PetalWidthCm'], label="Not Iris-setosa")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Support Vector Machine Classification")
plt.legend()
plt.show()
