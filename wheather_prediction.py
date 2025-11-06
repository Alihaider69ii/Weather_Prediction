import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv("weather_prediction_dataset.csv")

# List of places
places = ["BUDAPEST","DE_BILT","DRESDEN","DUSSELDORF","HEATHROW","KASSSEL",
          "LJUBLJANA","MAASTRICTH","MALMO","MONTELIMAR","MUENCHEN","OSLO",
          "PERPIGNAN","ROMA","SONNBLICK","STOCKHOLM","TOURS"]

print("Available places:", places)

# Ask user input
place = input("Enter place name from above list: ").strip().upper()
month = float(input("Enter month number (1-12): "))

# Build X and y dynamically based on place
X = data[['MONTH']]
y = data[[f"{place}_cloud_cover",
          f"{place}_humidity",
          f"{place}_pressure",
          f"{place}_global_radiation",
          f"{place}_precipitation",
          f"{place}_sunshine"]]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
prediction = model.predict([[month]])

# Evaluation
result = model.predict(X)
mae = mean_absolute_error(y, result)
mse = mean_squared_error(y, result)
rmse = np.sqrt(mse)

# Print results
print("\nModel evaluation:")
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

print(f"\nBased on month {int(month)} in {place}, predicted weather is:")
print(f"Cloud cover: {prediction[0][0]:.2f}")
print(f"Humidity: {prediction[0][1]:.2f}")
print(f"Pressure: {prediction[0][2]:.2f}")
print(f"Global radiation: {prediction[0][3]:.2f}")
print(f"Precipitation: {prediction[0][4]:.2f}")
print(f"Sunshine: {prediction[0][5]:.2f}")

