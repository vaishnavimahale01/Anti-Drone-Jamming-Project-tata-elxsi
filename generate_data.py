import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=100):
    # Generate synthetic sensor data (signal_strength, time_of_arrival, SNR)
    X = np.random.rand(n_samples, 3) * 100  # 3 features: Signal Strength, TOA, SNR
    
    # Generate synthetic jammer positions (x, y)
    y = np.column_stack((np.random.uniform(0, 100, n_samples), np.random.uniform(0, 100, n_samples)))
    
    return X, y

def save_data_to_csv(X, y, X_filename="X_data.csv", y_filename="y_data.csv"):
    # Save the generated data to CSV files
    X_df = pd.DataFrame(X, columns=["Signal Strength", "TOA", "SNR"])
    y_df = pd.DataFrame(y, columns=["Jammer X", "Jammer Y"])
    
    X_df.to_csv(X_filename, index=False)
    y_df.to_csv(y_filename, index=False)
    
    print(f"Data saved to '{X_filename}' and '{y_filename}'")

# Generate synthetic data and save to CSV
if __name__ == "__main__":
    X, y = generate_synthetic_data(100)  # Generate 100 samples
    save_data_to_csv(X, y)
