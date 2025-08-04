# pca_gene_expression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('data.csv')  # Replace with actual path if needed

# Separate labels and features
labels = df['label']
features = df.drop(columns=['label'])

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# Create DataFrame for PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['label'] = labels

# Plot PCA results
plt.figure(figsize=(10,7))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df, palette='tab10')
plt.title('PCA of Gene Expression Data')
plt.xlabel(f'PC1 - {pca.explained_variance_ratio_[0]:.2%} Variance')
plt.ylabel(f'PC2 - {pca.explained_variance_ratio_[1]:.2%} Variance')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_plot.png")
plt.show()