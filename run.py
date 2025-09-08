import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import random
import os

# Step 1: Load Data (Either from CSV or Generate Synthetic Data)
def load_data(input_option="synthetic"):
    if input_option == "synthetic":
        from generate_data import generate_synthetic_data
        # Generate synthetic data
        X_train, y_train = generate_synthetic_data(n_samples=100)
        X_test, y_test = generate_synthetic_data(n_samples=20)
    else:
        # Ask user for dataset file paths
        X_filename = input("Enter the path to the X data CSV file: ")
        y_filename = input("Enter the path to the y data CSV file: ")
        
        # Check if files exist
        if not os.path.exists(X_filename) or not os.path.exists(y_filename):
            print("Error: One or both of the files do not exist.")
            return None, None
        
        # Load data from CSV
        X_train = pd.read_csv(X_filename).values
        y_train = pd.read_csv(y_filename).values
        X_test, y_test = X_train[:20], y_train[:20]  # Simple split for testing
        
    return X_train, y_train, X_test, y_test

# Step 2: Train Random Forest Regressor for Jammer Position Estimation
def train_position_estimation_model(X_train, y_train):
    # Train a Random Forest model to estimate jammer positions
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 3: Evaluate the Model (Mean Squared Error)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return y_pred, mse

# Step 4: Save Results to a Text File
def save_results(y_pred, y_true, mse, file_name):
    with open(file_name, "w") as file:
        file.write("Jammer Position Estimation Results\n")
        file.write("===============================\n")
        
        # Save predicted positions
        file.write("\nPredicted Jammer Positions (x, y):\n")
        for i, position in enumerate(y_pred):
            file.write(f"Test Sample {i+1}: ({position[0]:.2f}, {position[1]:.2f})\n")
        
        # Save true positions for comparison
        file.write("\nTrue Jammer Positions (x, y):\n")
        for i, position in enumerate(y_true):
            file.write(f"Test Sample {i+1}: ({position[0]:.2f}, {position[1]:.2f})\n")
        
        # Save evaluation metrics
        file.write(f"\nMean Squared Error (MSE): {mse:.2f}\n")
        
        file.write("===============================\n")
        file.write("End of Results.\n")
    print(f"Results saved to '{file_name}'")

# Step 5: Simulate Reinforcement Learning for Mitigation Strategy (Simplified)
class RLAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((100, 100))  # Simplified Q-table for positions

    def act(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best action based on Q-table

    def update_q_table(self, state, action, reward, next_state):
        # Ensure next_state stays within bounds (0 to 99)
        next_state = np.clip(next_state, 0, 99)
        
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )

    def train(self, episodes=1000):
        total_rewards = 0
        for episode in range(episodes):
            state = random.randint(0, 99)  # Random initial state (position)
            action = self.act(state)
            reward = -1 if action == 0 else 1  # Reward based on action (simplified)
            next_state = state + random.choice([-1, 1])  # Next state (simplified)
            
            self.update_q_table(state, action, reward, next_state)
            total_rewards += reward
        return total_rewards

# Step 6: Main Execution
def main():
    input_option = input("Do you want to use synthetic data or input your own data? (synthetic/input): ").strip().lower()

    if input_option not in ["synthetic", "input"]:
        print("Invalid option. Please enter 'synthetic' or 'input'.")
        return
    
    # 1. Load data
    X_train, y_train, X_test, y_test = load_data(input_option=input_option)
    if X_train is None or y_train is None:
        return
    
    # 2. Train the position estimation model
    model = train_position_estimation_model(X_train, y_train)
    
    # 3. Evaluate the model
    y_pred, mse = evaluate_model(model, X_test, y_test)
    
    # 4. Save the results of the jammer position estimation
    save_results(y_pred, y_test, mse, "jammer_detection_results.txt")
    
    # 5. Train the RL agent for jammer mitigation (simplified example)
    agent = RLAgent()
    total_rewards = agent.train(episodes=1000)
    
    # Save RL agent's performance
    with open("rl_mitigation_results.txt", "w") as file:
        file.write("Reinforcement Learning - Jammer Mitigation Results\n")
        file.write("===============================\n")
        file.write(f"Total Rewards: {total_rewards}\n")
        file.write("===============================\n")
        file.write("End of Results.\n")
    print("RL Mitigation results saved to 'rl_mitigation_results.txt'")

# Run the main function
if __name__ == "__main__":
    main()
