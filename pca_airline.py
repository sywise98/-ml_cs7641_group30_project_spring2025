
from pca import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

class AirlinePCA(PCA):
    
    def __init__(self):
        super().__init__()
        
        
    def read_data(self, csv_file_name):
        df = pd.read_csv(csv_file_name)
        print(df.info())
        print(df.describe())
        print(df.isnull().sum())
        
        # print(df[:5])
        return df
    
    
    def clean_data(self, df):
        #* Drop duplicates
        df = df.drop_duplicates()
        
        #* Handle missing values
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # Fill categorical columns with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')

        #* Convert time features
        # Convert DepTime and ArrTime to minutes since midnight
        # Make sure DepTime and ArrTime are safe integers first
        df['DepTime'] = df['DepTime'].fillna(0).astype(float).astype(int)
        df['ArrTime'] = df['ArrTime'].fillna(0).astype(float).astype(int)

        # Then safely convert to minutes since midnight
        df['DepMinutes'] = df['DepTime'].astype(str).str.zfill(4).str[:2].astype(int) * 60 + \
                        df['DepTime'].astype(str).str.zfill(4).str[2:].astype(int)

        df['ArrMinutes'] = df['ArrTime'].astype(str).str.zfill(4).str[:2].astype(int) * 60 + \
                        df['ArrTime'].astype(str).str.zfill(4).str[2:].astype(int)

        # Drop the original columns
        df = df.drop(columns=['DepTime','ArrTime'])

        #* Encode categorical variables using one-hot encoding
        categorical_features_to_encode = ['UniqueCarrier', 'Origin', 'Dest']
        for feature in categorical_features_to_encode:
            if feature in df.columns:
                df = pd.get_dummies(df, columns=[feature], drop_first=True)

        #* Remove cancelled flights
        if 'Cancelled' in df.columns:
            df = df[df['Cancelled'] != 1]

        #* Drop irrelevant or unhelpful columns
        drop_cols = ['Year', 'FlightNum', 'TailNum', 'Cancelled', 'CancellationCode', 'Diverted']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        #* Remove rows with missing target
        df = df.dropna(subset=['ArrDelay'])

        #* Define features and target
        y = df['ArrDelay']
        X = df.drop('ArrDelay', axis=1)

        #* Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Data cleaning and splitting done!")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    
    def delay_viz(self, X_sample, y_sample):
        vmin = np.percentile(y_sample, 5)
        vmax = np.percentile(y_sample, 95)
        # Plot with color representing delay
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            X_sample[:, 1], X_sample[:, 2],     #! These features being visualized can be changed!
            c=y_sample,
            cmap='coolwarm',  # blue = early, red = late
            vmin=vmin, vmax=vmax,  # ← clip color range manually
            s=15,
            alpha=0.8
        )

        plt.colorbar(scatter, label='Arrival Delay (minutes)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Projection Colored by Arrival Delay')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # plt.scatter(X[:, 0], X[:, 1])
        # plt.title('PCA Projection')
        # plt.show()
        
    def airport_viz(self, X_sample, X_train, sample_indices):
        origin_iah_values = X_train.iloc[sample_indices]['Origin_IAH'].values
        # Convert Origin one-hot column to label
        # X_train_sample = X_train.loc[sample_indices]
        airport_labels = np.where(origin_iah_values == 1, 'IAH', 'HOU')

        # Create a color map
        color_map = {'IAH': 'blue', 'HOU': 'orange'}
        colors = [color_map[label] for label in airport_labels]

        plt.figure(figsize=(8,6))
        plt.scatter(X_sample[:, 0], X_sample[:, 1], c=colors, alpha=0.5)

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Projection Colored by Origin Airport")
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='IAH', markerfacecolor='blue', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='HOU', markerfacecolor='orange', markersize=8)
        ])
        plt.grid(True)
        plt.show()
    
    def inspect_weights(self, pca, X_train, pc_count):
        for i in range(pc_count):
            # Turn PC1 into a pandas Series for easier inspection
            pc_weights = pd.Series(pca.V[i], index=X_train.columns)

            # Sort by absolute contribution
            pc_top_features = pc_weights.abs().sort_values(ascending=False)
            print(f"Top features contributing to PC{i}:")
            print(pc_top_features.head(10))  # top 10


if "__main__" == __name__:
    pca = AirlinePCA()
    
    data_file_name = "hflights.csv"
    
    df = pca.read_data(csv_file_name=data_file_name)
    X_train, X_test, y_train_df, y_test_df = pca.clean_data(df)
    
    X_train_np = X_train.astype(float).to_numpy()
    X_test_np = X_test.astype(float).to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit training data to a model
    pca.fit(X_train_scaled)
    
    #! This can be flipped to grabbing specifc number of principal components (K) if needed
    # X_train_pca = pca.transform(X_train_scaled, K=3)
    # X_test_pca = pca.transform(X_test_scaled, K=3)
    X_train_pca = pca.transform_rv(X_train_scaled, retained_variance=0.90)
    X_test_pca = pca.transform_rv(X_test_scaled, retained_variance=0.90)
    
    print(f"X train pca shape: {X_train_pca.shape}")
    print(f"X test pca shape: {X_test_pca.shape}")
    
    sample_size = 5000
    # Randomly choose indices
    indices = np.random.choice(X_train_pca.shape[0], size=sample_size, replace=False)
    # Subset the data
    X_sample = X_train_pca[indices]
    y_sample = y_train_df.astype(float).to_numpy()[indices]
    pca.delay_viz(X_sample, y_sample)
    pca.airport_viz(X_sample, X_train, indices)
    pca.visualize(X=X_sample, y=y_sample, fig_title="Airline PCA Projection")
    
    # See which original features are weighted the most in the principal components
    pca.inspect_weights(pca, X_train, pc_count=X_train_pca.shape[1])
    
    # Convert PCA results back to DataFrames for saving
    X_train_pca_df = pd.DataFrame(X_train_pca)
    X_test_pca_df = pd.DataFrame(X_test_pca)

    # Match shapes to avoid index issues when reloading
    X_train.to_csv("X_train_cleaned.csv", index=False)    #! Uncomment these for just cleaned data w/ no PCA to save to csv
    X_test.to_csv("X_test_cleaned.csv", index=False)
    X_train_pca_df.to_csv("X_train_pca.csv", index=False)
    X_test_pca_df.to_csv("X_test_pca.csv", index=False)
    y_train_df.to_csv("y_train.csv", index=False)
    y_test_df.to_csv("y_test.csv", index=False)
    
    print(f"Data has been cleaned, transformed, and saved!")
