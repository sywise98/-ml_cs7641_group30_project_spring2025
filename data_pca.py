from pca import PCA
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

class DataPCA(PCA):
    
    def __init__(self):
        super().__init__()
        self.weather_pca = PCA()
        self.flight_pca = PCA()
        
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
        
        # Replace "N" with 0 and "B" with 1 in the cancelled_code column
        df['cancelled_code'] = np.where(df['cancelled_code'] == 'N', 0, 1)
        
       # Combine all features first
        all_features = [
            # Flight features
            "flight_number", "scheduled_elapsed_time", "cancelled_code",
            # Weather features
            "HourlyDryBulbTemperature_x", "HourlyPrecipitation_x",
            "HourlyStationPressure_x", "HourlyVisibility_x", "HourlyWindSpeed_x",
            "HourlyDryBulbTemperature_y", "HourlyPrecipitation_y",
            "HourlyStationPressure_y", "HourlyVisibility_y", "HourlyWindSpeed_y"
        ]
        
        # Clean features and target together
        X = df[all_features].dropna()
        y = pd.to_numeric(df['arrival_delay'], errors='coerce').loc[X.index].dropna()

        # Ensure aligned indices
        X = X.loc[y.index]
        
        # Single split for all data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Separate flight/weather features AFTER splitting
        flight_cols = ["flight_number", "scheduled_elapsed_time", "cancelled_code"]
        weather_cols = [col for col in all_features if col not in flight_cols]
        
        return (
            X_train[weather_cols], X_test[weather_cols],
            X_train[flight_cols], X_test[flight_cols],
            y_train, y_test
        )
        
    def fit_pca(self, data_type, X):
        """Train PCA for specific data type"""
        if data_type == "weather":
            self.weather_pca.fit(X)
        elif data_type == "flight":
            self.flight_pca.fit(X)
    
    def transform_pca(self, data_type, X, retained_variance):
        """Apply PCA transformation"""
        if data_type == "weather":
            return self.weather_pca.transform_rv(X, retained_variance)
        elif data_type == "flight":
            return self.flight_pca.transform_rv(X, retained_variance)
    
    def inspect_weights(self, data_type, X_train, pc_count):
        """Analyze PCA component weights for specific data type"""
        pca_instance = self.weather_pca if data_type == "weather" else self.flight_pca
        for i in range(pc_count):
            pc_weights = pd.Series(pca_instance.V[i], index=X_train.columns)
            pc_top_features = pc_weights.abs().sort_values(ascending=False)
            print(f"Top features contributing to {data_type} PC{i}:")
            print(pc_top_features.head(10))


            
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
    
            
if "__main__" == __name__:
    pca = DataPCA()
    
    current_directory = os.getcwd()
    data_file_name = os.path.join(current_directory, "archive", "05-2019.csv")
    
    df = pca.read_data(csv_file_name=data_file_name)
    X_weather_train, X_weather_test, X_flight_train, X_flight_test, y_train, y_test = pca.clean_data(df)
    
    X_weather_train_np = X_weather_train.astype(float).to_numpy()
    X_weather_test_np = X_weather_test.astype(float).to_numpy()
    X_flight_train_np = X_flight_train.astype(float).to_numpy()
    X_flight_test_np = X_flight_test.astype(float).to_numpy()

    scaler = StandardScaler()
    X_weather_train_scaled = scaler.fit_transform(X_weather_train)
    X_weather_test_scaled = scaler.transform(X_weather_test)
    X_flight_train_scaled = scaler.fit_transform(X_flight_train)
    X_flight_test_scaled = scaler.transform(X_flight_test)
    
    # Fit training data to a model
    pca.fit(X_weather_train_scaled)
    pca.fit(X_flight_train_scaled)
    
    # Fit and transform weather data
    pca.fit_pca("weather", X_weather_train_scaled)
    X_weather_train_pca = pca.transform_pca("weather", X_weather_train_scaled, 0.90)

    # Fit and transform flight data
    pca.fit_pca("flight", X_flight_train_scaled)
    X_flight_train_pca = pca.transform_pca("flight", X_flight_train_scaled, 0.90)
    
    print(f"X train weather pca shape: {X_weather_train_pca.shape}")
    # print(f"X test weather pca shape: {X_weather_test_pca.shape}")
    print(f"X train flight pca shape: {X_flight_train_pca.shape}")
    # print(f"X test flight pca shape: {X_flight_test_pca.shape}")
    
    sample_size = 5000
    # Randomly choose indices
    weather_indices = np.random.choice(X_weather_train_pca.shape[0], size=sample_size, replace=False)
    flight_indices  = np.random.choice(X_flight_train_pca.shape[0], size=sample_size, replace=False)
    
    
    # Subset the data
    X_weather_sample = X_weather_train_pca[weather_indices]
    X_flight_sample  = X_flight_train_pca[flight_indices]
    y_sample = y_train.astype(float).to_numpy()[flight_indices]
    
    pca.delay_viz(X_weather_sample, y_sample)
    pca.delay_viz(X_flight_sample, y_sample)
    
    pca.visualize(X=X_weather_sample, y=y_sample, fig_title="Weather PCA Projection")
    pca.visualize(X=X_flight_sample, y=y_sample, fig_title="Airline PCA Projection")
    
    # Fixed version:
    pca.inspect_weights("weather", X_weather_train, pc_count=X_weather_train_pca.shape[1])
    pca.inspect_weights("flight", X_flight_train, pc_count=X_flight_train_pca.shape[1])
    
    # Convert PCA results back to DataFrames for saving
    X_weather_train_pca_df = pd.DataFrame(X_weather_train_pca)
    # X_weather_test_pca_df = pd.DataFrame(X_weather_test_pca)
    X_flight_train_pca_df = pd.DataFrame(X_flight_train_pca)
    # X_flight_test_pca_df = pd.DataFrame(X_flight_test_pca)
    
    X_weather_train_pca_df.to_csv("X_weather_train_cleaned.csv", index=False)
    # X_flight_test.to_csv("X_flight_test_cleaned.csv", index=False)
    X_flight_train_pca_df.to_csv("X_flight_train_pca.csv", index=False)
    # X_flight_test_pca_df.to_csv("X_flight_test_pca.csv", index=False)
    
    # y_train.to_csv("y_train.csv", index=False)
    # y_test.to_csv("y_test.csv", index=False)
    
    print(f"Data has been cleaned, transformed, and saved!")