import sys
import json
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
from statsmodels.tools.sm_exceptions import ValueWarning
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load the data passed from PHP
input_data = pd.read_csv('C:/xampp/htdocs/FYH/fyh-source-code/cms/content/input_data.csv')

# Prepare the data
input_data['modified'] = pd.to_datetime(input_data['modified'])

# Drop rows where 'modified' is NaT
input_data.dropna(subset=['modified'], inplace=True)

# Remove rows with empty customer_id
input_data = input_data[input_data['customer_id'].notna()]

# Remove duplicates
input_data.drop_duplicates(inplace=True)

# Drop rows with missing values in these columns
input_data.dropna(subset=['product_id', 'product_name', 'qty', 'modified'], inplace=True)

# Remove rows where quantity is zero
db_data = input_data[input_data['qty'] > 0]

# Ensure data consistency, e.g., quantity and unit_price should be non-negative
db_data = db_data[db_data['qty'] >= 0]

# Handling outliers by capping at the 95th percentile
percentiles = db_data[['qty']].quantile(0.95)
db_data['qty'] = np.where(db_data['qty'] > percentiles['qty'], percentiles['qty'], db_data['qty'])

# Initialize the results list
forecast_results = []

# Process data for each customer and product
for (customer_id, product_id), group in db_data.groupby(['customer_id', 'product_id']):
    product_name = group['product_name'].iloc[0]
    
    # Set 'modified' as the index and ensure it is a proper DatetimeIndex
    group.set_index('modified', inplace=True)
    group.index = pd.to_datetime(group.index)

    # Resample quantity data by week
    time_series_data = group['qty'].resample('W').sum().fillna(0)

    # Ensure that there are enough data points for training and forecasting
    if len(time_series_data) < 1:
        print(f"Not enough data for customer {customer_id}, product {product_id}")
        continue

    # Load SARIMA model specific to this customer-product pair
    model_path = os.path.join(script_dir, f'forecasting_models/sarima_model_{customer_id}_{product_id}.pkl')
    
    try:
        # Load the SARIMA model if it exists
        sarima_model = joblib.load(model_path)
        
        # Forecast the next 10 periods starting from today
        steps = 10
        today = datetime.today()

        # Check if the model was fitted with a proper index
        try:
            forecast = sarima_model.get_forecast(steps=steps).predicted_mean
        except ValueWarning:
            # Manually handle the forecast if there's no proper index
            print(f"Model for customer {customer_id}, product {product_id} lacks proper index")
            forecast = sarima_model.get_forecast(steps=steps).predicted_mean
        
        # Prepare the forecast data for each customer and product
        for i, qty in enumerate(forecast):
            forecast_date = today + pd.DateOffset(weeks=i)
            forecast_results.append({
                'customer_id': int(customer_id),  # Convert to Python int
                'product_id': int(product_id),  # Convert to Python int
                'product_name': product_name,
                'date': forecast_date.strftime('%Y-%m-%d'),
                'quantity': max(0, round(qty))  # Ensure non-negative and rounded
            })
    
    except FileNotFoundError:
        continue
    except Exception as e:
        continue

# Return the forecast as JSON
print(json.dumps(forecast_results))
