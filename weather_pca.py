import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCA(object):

	def __init__(self):
		self.U = None
		self.S = None
		self.V = None
  
	def inspect_weights(self, pca, X_train, pc_count):
		for i in range(pc_count):
			# Turn PC1 into a pandas Series for easier inspection
			pc_weights = pd.Series(pca.V[i], index=X_train.columns)

			# Sort by absolute contribution
			pc_top_features = pc_weights.abs().sort_values(ascending=False)
			print(f"Top features contributing to PC{i}:")
			print(pc_top_features.head(10))  # top 10
   
	def check_plot(self, X_sample, y_sample):
		vmin = np.percentile(y_sample, 5)
		vmax = np.percentile(y_sample, 95)
		# Plot with color representing delay
		plt.figure(figsize=(10, 6))
		scatter = plt.scatter(
			X_sample[:, 1], X_sample[:, 2],
			c=y_sample,
			cmap='coolwarm',  # blue = early, red = late
			vmin=vmin, vmax=vmax,  # ← clip color range manually
			s=15,
			alpha=0.8
		)

		plt.colorbar(scatter, label='Unix Timestamp (Y-M-D H:M:S)')
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.title('PCA Projection Colored by Unix Timestamp')
		plt.grid(True)
		plt.tight_layout()
		plt.show()
		# plt.scatter(X[:, 0], X[:, 1])
		# plt.title('PCA Projection')
		# plt.show()

	def fit(self, X: np.ndarray) ->None:
		"""		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose

		Hint: np.linalg.svd by default returns the transpose of V
				Make sure you remember to first center your data by subtracting the mean of each feature.

		Args:
			X: (N,D) numpy array corresponding to a dataset

		Return:
			None

		Set:
			self.U: (N, min(N,D)) numpy array
			self.S: (min(N,D), ) numpy array
			self.V: (min(N,D), D) numpy array
		"""
		# Center the data
		self.mean = np.mean(X, axis=0)
		X_centered = X - self.mean

		# Perform SVD
		self.U, self.S, self.V = np.linalg.svd(X_centered, full_matrices=False)

	def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
		"""		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		by projecting onto the principal components.
		Utilize class members that were previously set in fit() method.

		Args:
			data: (N,D) numpy array corresponding to a dataset
			K: int value for number of columns to be kept

		Return:
			X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data

		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
		# Center the data using the mean from fit
		X_centered = data - self.mean

		# Project the centered data onto the first K principal components
		return X_centered @ self.V[:K].T

	def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
	) ->np.ndarray:
		"""		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.

		Args:
			data: (N,D) numpy array corresponding to a dataset
			retained_variance: float value for amount of variance to be retained

		Return:
			X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
					to be kept to ensure retained variance value is retained_variance

		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
		# Center the data using the mean from fit
		X_centered = data - self.mean

		# Calculate the cumulative explained variance ratio
		explained_variance_ratio = (self.S ** 2) / np.sum(self.S ** 2)
		cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

		# Find the number of components needed to retain the specified variance
		K = np.argmax(cumulative_variance_ratio >= retained_variance) + 1

		# Project the centered data onto the first K principal components
		return X_centered @ self.V[:K].T

	def get_V(self) ->np.ndarray:
		"""		
		Getter function for value of V
		"""
		return self.V

	def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
		"""		
		You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
		For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
		You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using matplotlib.

		Args:
			xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
			ytrain: (N,) numpy array, the true labels

		Return: None
		"""
		# Fit PCA to the data
		self.fit(X)

		# 2D plot with strongest two features
		X_2d = self.transform(X, K=2)
		plt.figure(figsize=(10, 8))
		scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
		plt.colorbar(scatter)
		plt.title(f'{fig_title} - 2D PCA (Strongest Features)')
		plt.xlabel('First Principal Component')
		plt.ylabel('Second Principal Component')
		plt.show()

		# 3D plot with strongest three features
		X_3d = self.transform(X, K=3)
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')
		scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis')
		fig.colorbar(scatter)
		ax.set_title(f'{fig_title} - 3D PCA (Strongest Features)')
		ax.set_xlabel('First Principal Component')
		ax.set_ylabel('Second Principal Component')
		ax.set_zlabel('Third Principal Component')
		plt.show()

		# 2D plot with two random features
		random_features = np.random.choice(X.shape[1], 2, replace=False)
		plt.figure(figsize=(10, 8))
		scatter = plt.scatter(X[:, random_features[0]], X[:, random_features[1]], c=y, cmap='viridis')
		plt.colorbar(scatter)
		plt.title(f'{fig_title} - 2D Random Features')
		plt.xlabel(f'Random Feature {random_features[0]}')
		plt.ylabel(f'Random Feature {random_features[1]}')
		plt.show()
