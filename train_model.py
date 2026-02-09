import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("Mall_Customers.csv")

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Save model and scaler
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save clustered data
df['Cluster'] = kmeans.labels_
df.to_csv("clustered_mall_customers.csv", index=False)

print("✅ Model, scaler, and clustered data saved successfully")
