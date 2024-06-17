
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('house_price_dataset_original_v2_cleaned.csv')

# Define target variable and features
y = data['property_value']
x = data[['land_size_sqm', 
          'house_size_sqm', 
          'no_of_rooms', 
          'no_of_bathrooms', 
          'large_living_room', 
          'parking_space', 
          'front_garden', 
          'swimming_pool', 
          'distance_to_school', 
          'wall_fence', 
          'house_age', 
          'water_front', 
          'distance_to_supermarket_km', 
          'crime_rate_index', 'room_size']]


from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# Train the model
from sklearn.linear_model import LinearRegression
alg1 = LinearRegression()
alg1.fit(x_train, y_train)

# Visualize the results
y_pred = alg1.predict(x_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='Black', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='Blue', linewidth=3)
plt.xlabel('Actual Property Value')
plt.ylabel('Predicted Property Value')
plt.title('Actual vs Predicted House Price')
plt.grid(True)
plt.show()

print(f"Training Accuracy: {alg1.score(x_train, y_train)}")

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = [
            data.get('land_size_sqm', '0'),
            data.get('house_size_sqm', '0'),
            data.get('no_of_rooms', '0'),
            data.get('no_of_bathrooms', '0'),
            data.get('large_living_room', '0'),
            data.get('parking_space', '0'),
            data.get('front_garden', '0'),
            data.get('swimming_pool', '0'),
            data.get('distance_to_school', '0'),
            data.get('wall_fence', '0'),
            data.get('house_age', '0'),
            data.get('water_front', '0'),
            data.get('distance_to_supermarket_km', '0'),
            data.get('crime_rate_index', '0'),
            data.get('room_size', '0')
        ]
        
        # Clean and convert features to numpy array
        features = [float(value.strip()) if value.strip() else 0.0 for value in features]
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = alg1.predict(features)

        # Render the result
        return render_template('index.html', prediction_text=f'Predicted House Price: {prediction[0]:.2f}', request=request)
    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: {e}', request=request)
    except Exception as e:
        return render_template('index.html', prediction_text=f'An unexpected error occurred: {e}', request=request)

if __name__ == "__main__":
    app.run(debug=True)

