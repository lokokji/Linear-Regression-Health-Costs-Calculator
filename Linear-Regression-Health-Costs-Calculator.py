# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Step 2: Load the dataset
url = 'https://your-dataset-url'  # Replace with your actual dataset URL or file path
data = pd.read_csv(url)

# Step 3: Preprocess the data
# Convert categorical columns to numbers using LabelEncoder
categorical_columns = ['sex', 'region', 'smoker', 'children']  # Modify based on actual categorical columns
encoder = LabelEncoder()

for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

# Step 4: Split the data into training and testing sets
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)

# Step 5: Pop the "expenses" column to create train_labels and test_labels
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# Step 6: Create a Regression Model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[train_dataset.shape[1]]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression (single value)
])

# Step 7: Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Step 8: Train the model
history = model.fit(train_dataset, train_labels, epochs=50, validation_split=0.2)

# Step 9: Evaluate the model using the test set
test_loss = model.evaluate(test_dataset, test_labels)
print(f"Test Loss (Mean Absolute Error): {test_loss}")

# Step 10: Predict using the model and visualize the results
predictions = model.predict(test_dataset).flatten()

# Plot the actual vs predicted expenses
plt.scatter(test_labels, predictions)
plt.xlabel('Actual Expenses')
plt.ylabel('Predicted Expenses')
plt.title('Actual vs Predicted Healthcare Expenses')
plt.show()

# Check if the model is under the required MAE (3500)
if test_loss < 3500:
    print("Model passed with a Mean Absolute Error under 3500.")
else:
    print("Model did not pass. Consider improving the model.")
