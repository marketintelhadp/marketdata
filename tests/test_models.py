# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse  # Importing argparse

# Load the dataset
data = pd.read_csv(r'D:\Git Projects\Price_forecasting_project\Agricultural-Price-Intelligence\data\raw\processed\Sopore\Delicious_A_dataset.csv')

#data = pd.read_csv(r'D:\ML Repositories\Price_forecasting_project\data\raw\processed\Narwal\Razakwadi_dataset.csv')
# Ensure proper datetime format for models requiring 'ds'
data = data.rename(columns={"Date": "ds", "Avg Price (per kg)": "y"})
data['ds'] = pd.to_datetime(data['ds'])
# Filter for available data (Mask=1) for SARIMA and Prophet
available_data = data[data['Mask'] == 1].copy()
available_data.reset_index(inplace=True)

# Split data for training and testing
train_data = available_data[available_data['ds'] < '2024-08-01']
test_data = available_data[available_data['ds'] >= '2024-08-01']

# Scale the target variable
scaler = StandardScaler()
train_data['y_scaled'] = scaler.fit_transform(train_data[['y']])
test_data['y_scaled'] = scaler.transform(test_data[['y']])

# Function to reverse scaling
def reverse_scaling(scaled_values, scaler):
    return scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()

# SARIMA Model
def sarima_model(train_data, test_data):
    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)
    P_values = range(0, 2)
    D_values = range(0, 2)
    Q_values = range(0, 2)
    seasonal_periods = [12]

    param_grid = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_periods))
    results = []

    for param in param_grid:
        try:
            p, d, q, P, D, Q, S = param
            sarima_model_inst = SARIMAX(train_data['y_scaled'], order=(p, d, q), seasonal_order=(P, D, Q, S))
            sarima_results = sarima_model_inst.fit(disp=False)
            results.append((param, sarima_results.aic))
        except Exception as e:
            continue

    if results:
        results_df = pd.DataFrame(results, columns=["Params", "AIC"])
        best_params = results_df.sort_values(by="AIC").iloc[0]
        print("\nBest Parameters for SARIMA:", best_params)

        best_p, best_d, best_q, best_P, best_D, best_Q, best_S = best_params[0]
        sarima_model_inst = SARIMAX(train_data['y_scaled'], order=(best_p, best_d, best_q), seasonal_order=(best_P, best_D, best_Q, best_S))
        sarima_results = sarima_model_inst.fit(disp=False)
        sarima_forecast = sarima_results.get_forecast(steps=len(test_data))
        return reverse_scaling(sarima_forecast.predicted_mean.values, scaler)  # Include seq_length
    else:
        print("No valid SARIMA models were successfully fitted.")
        return None, 0  # Return None and a default seq_length

# Prophet Model
def prophet_model(train_data, test_data):
    prophet_model_inst = Prophet()
    # Fit the model using the original 'y' values
    prophet_model_inst.fit(train_data[['ds', 'y']])  # Use 'y' instead of 'y_scaled'
    future = prophet_model_inst.make_future_dataframe(periods=len(test_data))
    prophet_forecast = prophet_model_inst.predict(future)
    # Return the predictions for the test period
    return prophet_forecast['yhat'][-len(test_data):]  # No need to reverse scale here

# Function to create lagged features
def create_lagged_features(data, max_lag):
    data = data.copy().reset_index(drop=True)
    for lag in range(1, max_lag + 1):
        data[f'y_lag{lag}'] = data['y_scaled'].shift(lag)
    # Drop any rows with NA values (ensuring all columns are aligned)
    return data.dropna().reset_index(drop=True)

# Function to create sequences for LSTM and Transformer
def create_sequences(data, seq_length):
    if len(data) <= seq_length:
        print("Error: Data length must be greater than the sequence length.")
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])  # Target value

    X, y = np.array(X), np.array(y)
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

# Function to split data into training and validation (for tuning only)
def train_val_split(data, split_date):
    train_subset = data[data['ds'] < split_date]
    val_subset = data[data['ds'] >= split_date]
    return train_subset, val_subset

# Function to find optimal lags for RF and XGBoost
def find_best_lags(model, train_data, val_data, features):
    best_lags, best_mse = 0, float('inf')
    for num_lags in range(1, len(features) + 1):
        lag_features = features[:num_lags]
        print("Train features shape:", train_data[lag_features].shape, "Train target shape:", train_data['y_scaled'].shape)
        print("Val features shape:", val_data[lag_features].shape, "Val target shape:", val_data['y_scaled'].shape)
        model.fit(train_data[lag_features], train_data['y_scaled'])  # Use scaled y
        val_predictions = model.predict(val_data[lag_features])
        mse = mean_squared_error(val_data['y_scaled'], val_predictions)  # Use scaled y
        if mse < best_mse:
            best_mse, best_lags = mse, num_lags 
    return best_lags, best_mse

# Function to find optimal sequence length for LSTM and Transformer
def find_best_seq_length(train_data, max_seq_length):
    best_length, best_mse = 0, float('inf')

    for seq_length in range(1, max_seq_length + 1):
        X, y = create_sequences(train_data['y_scaled'].values, seq_length)
        if len(X) == 0:
            continue

        # Define simple LSTM model
        model = Sequential([
            LSTM(100, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X.reshape((-1, seq_length, 1)), y, epochs=50, batch_size=32, verbose=0)
        predictions = model.predict(X.reshape((-1, seq_length, 1)))
        mse = mean_squared_error(y, predictions)

        if mse < best_mse:
            best_mse, best_length = mse, seq_length

    return best_length

# RANDOM FOREST MODEL
def random_forest_model(train_data, test_data):
    max_lag = 30
    # Split train_data into training and validation using a cutoff (e.g., '2023-08-01')
    train_subset, val_subset = train_val_split(train_data, '2023-08-01')
    test_subset = test_data  # Use externally provided test_data

    # Create lagged features AFTER splitting
    train_subset = create_lagged_features(train_subset, max_lag)
    val_subset = create_lagged_features(val_subset, max_lag)
    test_subset = create_lagged_features(test_subset, max_lag)

    # Find Best Lags
    features = [f'y_lag{i}' for i in range(1, max_lag + 1)]
    rf_model_inst = RandomForestRegressor(n_estimators=100, random_state=42)
    best_lags_rf, _ = find_best_lags(rf_model_inst, train_subset, val_subset, features)

    # Train on Selected Features
    final_rf_features = features[:best_lags_rf]
    rf_model_inst.fit(train_subset[final_rf_features], train_subset['y_scaled'])

    # Predict & Reverse Scaling
    rf_predictions_scaled = rf_model_inst.predict(test_subset[final_rf_features])
    return reverse_scaling(rf_predictions_scaled, scaler)

# XGBOOST MODEL
def xgboost_model(train_data, test_data):
    max_lag = 30 
    train_subset, val_subset = train_val_split(train_data, '2023-08-01')
    test_subset = test_data

    # Create lagged features AFTER Splitting
    train_subset = create_lagged_features(train_subset, max_lag)
    val_subset = create_lagged_features(val_subset, max_lag)
    test_subset = create_lagged_features(test_subset, max_lag)

    # Find Best Lags
    features = [f'y_lag{i}' for i in range(1, max_lag + 1)]
    xgb_model_inst = XGBRegressor(n_estimators=100, random_state=42)
    best_lags_xgb, _ = find_best_lags(xgb_model_inst, train_subset, val_subset, features)

    # Train on Selected Features
    final_xgb_features = features[:best_lags_xgb]
    xgb_model_inst.fit(train_subset[final_xgb_features], train_subset['y_scaled'])

    # Predict & Reverse Scaling
    xgb_predictions_scaled = xgb_model_inst.predict(test_subset[final_xgb_features])
    return reverse_scaling(xgb_predictions_scaled, scaler)

# LSTM MODEL
def lstm_model(train_data, test_data):
    max_seq_length = 30
    # Split train_data into training and validation
    train_subset, val_subset = train_val_split(train_data, '2023-08-01')
    test_subset = test_data  # Use externally provided test_data

    # Create lagged features AFTER Splitting
    train_subset = create_lagged_features(train_subset, max_seq_length)
    val_subset = create_lagged_features(val_subset, max_seq_length)
    test_subset = create_lagged_features(test_subset, max_seq_length)

    # Find Best Sequence Length
    seq_length = find_best_seq_length(train_subset, 5)

    # Create Sequences
    X_train, y_train = create_sequences(train_subset['y_scaled'].values, seq_length)
    X_val, y_val = create_sequences(val_subset['y_scaled'].values, seq_length)
    X_test, y_test = create_sequences(test_subset['y_scaled'].values, seq_length)

    if len(X_train) == 0 or len(X_test) == 0:
        print("No sequences created for training or testing.")
        return np.zeros(len(test_data)), seq_length

    # Define & Train LSTM
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train.reshape((-1, seq_length, 1)), y_train,
              validation_data=(X_val.reshape((-1, seq_length, 1)), y_val),
              epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])

    # Predict & Reverse Scaling
    lstm_pred_scaled = model.predict(X_test.reshape((-1, seq_length, 1)))
    lstm_pred = reverse_scaling(lstm_pred_scaled.flatten(), scaler)
    return lstm_pred, seq_length

# Custom Dataset for PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TRANSFORMER MODEL
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embed_dim)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        return self.fc_out(x[:, -1, :])  # Output only the last time step

def transformer_model(train_data, test_data):
    max_seq_length = 30
    # Split train_data into training and validation
    train_subset, val_subset = train_val_split(train_data, '2023-08-01')
    test_subset = test_data  # Use externally provided test_data

    # Create lagged features AFTER Splitting
    train_subset = create_lagged_features(train_subset, max_seq_length)
    val_subset = create_lagged_features(val_subset, max_seq_length)
    test_subset = create_lagged_features(test_subset, max_seq_length)

    # Find Best Sequence Length
    seq_length = find_best_seq_length(train_subset, 5)

    # Create Sequences
    X_train, y_train = create_sequences(train_subset['y_scaled'].values, seq_length)
    X_val, y_val = create_sequences(val_subset['y_scaled'].values, seq_length)
    X_test, y_test = create_sequences(test_subset['y_scaled'].values, seq_length)

    if len(X_train) == 0 or len(X_test) == 0:
        print("No sequences created for training or testing.")
        return np.zeros(len(test_data)), seq_length

    # Create Datasets & Dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define the Transformer Model
    transformer = TransformerModel(input_dim=1, embed_dim=32, num_heads=4, ff_dim=128, num_layers=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

    # Training with Validation & Early Stopping
    best_val_loss = float("inf")
    patience = 10  
    epochs_no_improve = 0

    for epoch in range(100):
        transformer.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = transformer(batch_X.unsqueeze(-1))
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        transformer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = transformer(batch_X.unsqueeze(-1))
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    # Predictions on Test Data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    transformer.eval()
    with torch.no_grad():
        transformer_pred_scaled = transformer(X_test_tensor).detach().numpy()

    # Reverse Scaling & Align Predictions
    transformer_pred = reverse_scaling(transformer_pred_scaled, scaler)
    transformer_pred_aligned = transformer_pred[test_subset['Mask'].iloc[:len(transformer_pred)] == 1]
    return transformer_pred_aligned, seq_length

import os
# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

# Function to plot actual vs predicted and save the plot
def save_plot(y_true, y_pred, model_name, dates):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_true, label='Actual', color='blue')
    plt.plot(dates[:len(y_pred)], y_pred, label='Predicted', color='orange')  # Adjusted to match lengths
    plt.title(f'Actual vs Predicted for {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    # Save the plot in the results directory
    plot_dir = os.path.join("model_results","Sopore","Delicious_A", model_name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{model_name}_actual_vs_predicted.png")
    plt.savefig(plot_path)
    plt.close()

# Main function to run all models
def main():
    parser = argparse.ArgumentParser(description='Run specific models for price forecasting.')
    parser.add_argument('--model', type=str, choices=['sarima', 'prophet', 'rf', 'xgb', 'lstm', 'transformer'], 
                        help='Specify which model to run.')
    args = parser.parse_args()
    
    print(f"Selected model: {args.model}")  # Debugging information
    models_to_run = {
        'sarima': sarima_model,
        'prophet': prophet_model,
        'rf': random_forest_model,
        'xgb': xgboost_model,
        'lstm': lstm_model,
        'transformer': transformer_model 
    }

    if args.model:
        if args.model in models_to_run:
            try:
                pred = models_to_run[args.model](train_data, test_data)  # Only get predictions
                if pred is None or (isinstance(pred, tuple) and len(pred) == 0) or (not isinstance(pred, tuple) and len(pred) == 0):
                    print(f"{args.model.upper()} did not return valid predictions.")
                    return
                
                if args.model in ['lstm', 'transformer']:
                    pred, seq_length = models_to_run[args.model](train_data, test_data)
                    y_true = test_data['y'].values[test_data['Mask'] == 1][-len(pred):]
                    mse, mae = calculate_metrics(y_true, pred)
                    print(f"{args.model.upper()} Predictions: {pred}")
                    print(f"MSE: {mse}, MAE: {mae}")
                    dates = test_data['ds'].values[test_data['Mask'] == 1][-len(pred):]
                    save_plot(y_true, pred, args.model, dates)
                else:
                    y_true = test_data['y'].values[test_data['Mask'] == 1]
                    mse, mae = calculate_metrics(y_true, pred)
                    print(f"{args.model.upper()} Predictions: {pred}")
                    print(f"MSE: {mse}, MAE: {mae}")
                    dates = test_data['ds'].values[test_data['Mask'] == 1]
                    save_plot(y_true, pred, args.model, dates)

                result_dir = os.path.join("model_results","Sopore","Delicious_A", args.model)
                os.makedirs(result_dir, exist_ok=True)
                result_path = os.path.join(result_dir, f"{args.model}_results.txt")
                with open(result_path, "w") as f:
                    f.write(f"Model: {args.model.upper()}\n")
                    f.write(f"MSE: {mse}\n")
                    f.write(f"MAE: {mae}\n")
                    f.write("Predictions:\n")
                    f.write("\n".join(map(str, pred)))
                    
            except Exception as e:
                print(f"Error running {args.model.upper()}: {e}")
        else:
            print("Invalid model specified.")
    else:
        for model_name, model_func in models_to_run.items():
            try:
                pred = model_func(train_data, test_data)  # Only get predictions
                if pred is None or (isinstance(pred, tuple) and len(pred) == 0) or (not isinstance(pred, tuple) and len(pred) == 0):
                    print(f"{model_name.upper()} did not return valid predictions.")
                    continue
                if model_name in ['lstm', 'transformer']:
                    pred, seq_length = models_to_run[args.model](train_data, test_data)
                    y_true = test_data['y'].values[test_data['Mask'] == 1][-len(pred):]
                    mse, mae = calculate_metrics(y_true, pred)
                    print(f"{model_name.upper()} Predictions: {pred}")
                    print(f"MSE: {mse}, MAE: {mae}")
                    dates = test_data['ds'].values[test_data['Mask'] == 1][-len(pred):]
                    save_plot(y_true, pred, model_name, dates)
                else:
                    y_true = test_data['y'].values[test_data['Mask'] == 1]
                    mse, mae = calculate_metrics(y_true, pred)
                    print(f"{model_name.upper()} Predictions: {pred}")
                    print(f"MSE: {mse}, MAE: {mae}")
                    dates = test_data['ds'].values[test_data['Mask'] == 1]
                    save_plot(y_true, pred, model_name, dates)

                result_dir = os.path.join("model_results","Sopore", model_name)
                os.makedirs(result_dir, exist_ok=True)
                result_path = os.path.join(result_dir, f"{model_name}_results.txt")
                with open(result_path, "w") as f:
                    f.write(f"Model: {model_name.upper()}\n")
                    f.write(f"MSE: {mse}\n")
                    f.write(f"MAE: {mae}\n")
                    f.write("Predictions:\n")
                    f.write("\n".join(map(str, pred)))
            except Exception as e:
                print(f"Error running {model_name.upper()}: {e}")

if __name__ == "__main__":
    main()
