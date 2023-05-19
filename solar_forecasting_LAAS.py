# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:16:41 2023

@author: Gabriel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from time import time

# Define constants
n_steps_in = 24  # Number of input time steps
n_steps_out = 24  # Number of output time steps
n_features = 1  # Number of input features (i.e., number of columns in the input data)
n_units = 24  # Number of LSTM units in each layer
epochs = 100  # Number of epochs for training
batch_size = 64  # Batch size for training
patience = 10
# validation_split = 0.1  # Fraction of training data to use for validation

# Step 1: Load and preprocess the data
# Define a function to read and preprocess each file
def load_file(file_path):
    # Load the file as a dataframe
    df = pd.read_csv(file_path, header=None, index_col=False, usecols=[1, 2])
    # Convert the "Timestamp" column to a datetime object and set it as the index
    df["Timestamp"] = pd.to_datetime(df[1])
    df = df.set_index("Timestamp")
    # Resample the data to hourly frequency by taking the mean of each hour
    df = df.resample("H").mean()
    # Drop any rows with missing values
    df = df.dropna()
    df.columns = ['GHI']
    # Return the preprocessed dataframe
    return df

# Load and preprocess each file
file1 = 'data_complete/PHV.KIMO.PYRANO.CR100_1.MES1_1_2017_au_31_12_2017.csv'
file2 = 'data_complete/PHV.KIMO.PYRANO.CR100_1.MES1_1_2018_au_31_12_2018.csv'
file3 = 'data_complete/PHV.KIMO.PYRANO.CR100_1.MES1_1_2019_au_31_12_2019.csv'
file4 = 'data_complete/PHV.KIMO.PYRANO.CR100_1.MES1_1_2020_au_31_12_2020.csv'
file5 = 'data_complete/PHV.KIMO.PYRANO.CR100_1.MES1_1_2021_au_31_12_2021.csv'
file_paths = [file1, file2, file3, file4, file5]
dfs = [load_file(file_path) for file_path in file_paths]

# Concatenate the dataframes into a single dataframe
data = pd.concat(dfs)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data.values)

# Define the input and output sequences
X, y = [], []
for i in range(len(data_normalized)):
    # Find the end of the input and output sequences
    end_input = i + n_steps_in
    end_output = end_input + n_steps_out
    # Check if we have enough data for the input and output sequences
    if end_output >= len(data_normalized):
        break
    # Append the input and output sequences to X and y
    X.append(data_normalized[i:end_input])
    y.append(data_normalized[end_input:end_output])
X = np.array(X)
y = np.array(y)

# Split the data into training, validation, and test sets
X_train, y_train = X[:26280], y[:26280]  # First 3 years of data
X_valid, y_valid = X[26280:35040], y[26280:35040] # Fourth year of data
X_test, y_test = X[35040:], y[35040:] # Last year of data

# Step 2: Define and train the model
model = Sequential()

# Encoder
model.add(LSTM(n_units, activation="tanh", input_shape=(n_steps_in, n_features), return_sequences=True))
model.add(LSTM(n_units, activation="tanh", return_sequences=False))
model.add(RepeatVector(n_steps_out))

# Decoder
model.add(LSTM(n_units, activation="tanh", return_sequences=True))
model.add(LSTM(n_units, activation="tanh", return_sequences=True))
model.add(TimeDistributed(Dense(n_features))) # Output layer

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Train the model
start = time()
early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_valid, y_valid), callbacks=[early_stopping])
print("The model took %.2f seconds to train."% (time() - start)) 

print(model.summary())

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)), loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)), val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    # plt.axis([1, len(history.epoch), 0.1, 0.15])
    # plt.axis([1, len(history.epoch), min(history.history["val_loss"]), max(history.history["val_loss"])])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# Step 3: Evaluate the model on the test set
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred[y_pred<0] = 0

# Reshape the test and predicted output to match the desired shape
y_test = y_test.reshape((-1, n_steps_out))
y_pred = y_pred.reshape((-1, n_steps_out))

# Calculate the RMSE, MAE, and R squared for the test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R^2: {r2:.4f}")

# # Example to predict a new dataset
# # Assuming you have a new dataset of 24 hours stored in a variable called "new_data"

# # Normalize the new data using the same scaler
# # new_data_normalized = scaler.transform(new_data.reshape(-1, 1)).reshape(1, n_steps_in, 1)
# new_data_normalized = new_data.reshape(1, n_steps_in, 1)

# # Make predictions using the trained model
# new_data_predictions = model.predict(new_data_normalized)

# # Rescale the predictions back to the original scale
# new_data_predictions_rescaled = scaler.inverse_transform(new_data_predictions.reshape(-1, 1)).reshape(-1)

# # print("Predicted values:")
# # print(new_data_predictions_rescaled)

# Save the trained model
path = 'C:/Users/Luis Felipe Giraldo/OneDrive - Universidad de los andes/Doctorado/LAAS/Optimization/gabriel/models/'
# model.save("models/lstm_model.h5")
lstm_json_config = model.to_json()
with open(path+"lstm_model_MM.json", "w") as json_file:
    json_file.write(lstm_json_config)
# serialize weights to HDF5
model.save_weights(path+"lstm_weights_MM.h5")
print("Saved model to Disk")

# Example to load the saved model
# from tensorflow.keras.models import load_model
from tensorflow import keras

# Load the saved model
with open(path+'lstm_model.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights(path+'lstm_weights.h5')


