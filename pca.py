import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCA(object):

	def __init__(self):
		self.U = None
		self.S = None
		self.V = None
		self.mean = None

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
		self.mean = np.mean(X, axis=0)
		centered = X - self.mean
		self.U, self.S, self.V = np.linalg.svd(centered, full_matrices=False)

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
		# self.fit(data)
		# print(data.shape)
		# print(K)
		centered = data - self.mean
		X_new = centered @ self.V[:K, :].T
		# print(X_new.shape)
		return X_new

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
		# self.fit(data)
		# print(self.V)
		variance = np.square(self.S)
		# print(variance)
		total_var = np.sum(variance)
		cumul_var_ratio = np.cumsum(variance) / total_var
		# print(cumul_var_ratio)
		K = np.searchsorted(cumul_var_ratio, retained_variance) + 1		# +1 is to ensure it includes the variance element that would be greater than retained_variance
		print(f"K for transformed data: {K}")
		centered = data - self.mean
		X_new = centered @ self.V[:K, :].T
		return X_new

	def get_V(self) ->np.ndarray:
		"""		
		Getter function for value of V
		"""
		raise NotImplementedError

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
		# print(fig_title)
		
		# print(y[:10])
		y = y.astype(int)
		# unique_label_count = len(np.unique(y))
		# color_selection = np.array(['blue', 'green', 'red', 'purple', 'orange'])
		# colors = color_selection[y]
		# num_points = y.shape[0]
		# dot_sizes = np.full(num_points, 20)
  
		# X_new = self.transform(X, K=2)
		x_plot = X[:, 0]
		y_plot = X[:, 1]
		vmin = np.percentile(y, 5)
		vmax = np.percentile(y, 95)
		scatter = plt.scatter(
            x_plot, y_plot,
            c=y,
            cmap='coolwarm',  # blue = early, red = late
            vmin=vmin, vmax=vmax,  # ← clip color range manually
            s=15,
            alpha=0.8
        )
		plt.colorbar(scatter, label='Arrival Delay (minutes)')		
		# plt.scatter(x_plot, y_plot, c=y, cmap='coolwarm', s=20)
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title(f"{fig_title} - 2D PCA Reduction")
		plt.show()
  
		# X_new = self.transform(X, K=3)
		x_plot = X[:, 0]
		y_plot = X[:, 1]
		z_plot = X[:, 2]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(x_plot, y_plot, z_plot, c=y, cmap='coolwarm', s=20)
		ax.set_xlabel('Feature 1')
		ax.set_ylabel('Feature 2')
		ax.set_zlabel('Feature 3')
		ax.set_title(f"{fig_title} - 3D PCA Reduction")
		plt.show()
  
		# X_new = self.transform(X, K=2)
		feature_idxs = np.random.choice(X.shape[1], size=2, replace=False)
		x_plot = X[:, feature_idxs[0]]
		y_plot = X[:, feature_idxs[1]]
		plt.scatter(x_plot, y_plot, c=y, cmap='coolwarm', s=20)
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title(f"{fig_title} - Randomly Selected Features")
		plt.show()


