# Jammer Detection and Mitigation Project

## Description
This project aims to estimate the position of a drone jammer using machine learning and mitigate its effects using reinforcement learning (RL). The user can either use synthetic data or provide their own dataset for training and evaluation.

## Features
1. **Synthetic Data Generation**: Option to generate synthetic data using a predefined data generation function.
2. **User Input Data**: Option to input your own dataset in CSV format for training the model.
3. **Position Estimation**: Uses a Random Forest Regressor to estimate the position of the jammer.
4. **Jammer Mitigation**: A simple RL agent simulates the mitigation of jammer effects based on the trained model.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Generate Synthetic Data**: Run the `generate_data.py` file to generate synthetic sensor data.
   ```bash
   python generate_data.py
   ```

2. **Run the Model**: Execute the `run.py` file to start training the model:
   ```bash
   python run.py
   ```

   - The program will prompt you to select between synthetic data or input data:
     - If you select **synthetic**, it will use the generated data.
     - If you select **input**, you will need to provide the path to the `X_data.csv` and `y_data.csv` files.

## Files
- `generate_data.py`: Generates and saves synthetic data.
- `run.py`: Main script that loads data, trains the model, and performs jammer mitigation.
- `jammer_detection_results.txt`: File where the jammer position estimation results are saved.
- `rl_mitigation_results.txt`: File where the reinforcement learning results for jammer mitigation are saved.

## Results
- The **jammer position** is estimated and saved to `jammer_detection_results.txt`.
- The **reinforcement learning** results (total rewards) are saved to `rl_mitigation_results.txt`.

## Dependencies
- `numpy`
- `pandas`
- `scikit-learn`

To install these dependencies, you can use:
```bash
pip install numpy pandas scikit-learn
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
