import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Step 1: Load dataset
df = pd.read_csv("Train.csv")

# Step 2: Select numerical features
X = df[['Age', 'Work_Experience']]

# Step 3: Handle missing values (important for EM)
X = X.dropna()

# Step 4: Apply EM Algorithm using Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# Step 5: Predict cluster labels
labels = gmm.predict(X)

# Step 6: Visualize clusters
plt.figure()
plt.scatter(X['Age'], X['Work_Experience'], c=labels)
plt.xlabel("Age")
plt.ylabel("Work Experience")
plt.title("Expectation–Maximization (EM) Clustering")
plt.show()
