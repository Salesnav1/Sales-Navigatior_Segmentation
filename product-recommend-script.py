import sys
import json
import pandas as pd
import joblib
import numpy as np
import os
from surprise import KNNBasic, Dataset, Reader

# Load the KNN model
script_dir = os.path.dirname(os.path.realpath(__file__))
knn_model = joblib.load(os.path.join(script_dir, 'product_recommendation.pkl'))

# Load the input data from a CSV file
input_data = pd.read_csv('C:/xampp/htdocs/FYH/fyh-source-code/cms/content/input_data.csv')

# Log the input data for debugging
# with open('C:/xampp/htdocs/FYH/fyh-source-code/cms/content/input_data_log.txt', 'w') as f:
#     f.write(input_data.head().to_string())

# Normalize the quantity
input_data['quantity_normalized'] = (input_data['qty'] - input_data['qty'].mean()) / input_data['qty'].std()

# Convert data to long format for Surprise
long_format_data = input_data[['customer_id', 'product_id', 'qty']].dropna()
data = Dataset.load_from_df(long_format_data[['customer_id', 'product_id', 'qty']], Reader(rating_scale=(0, long_format_data['qty'].max())))

# Get customer ID from command line arguments
if len(sys.argv) > 1:
    customer_id = int(sys.argv[1])
else:
    print(json.dumps([]))
    sys.exit()

# Generate top 5 product recommendations for the selected customer
recommendations = []
trainset = knn_model.trainset

# Check if the customer exists in the training set
try:
    customer_inner_id = trainset.to_inner_uid(customer_id)
    customer_ratings = trainset.ur[customer_inner_id]
except ValueError:
    print(json.dumps([]))  # Customer not found
    sys.exit()

# Get the items already purchased by the customer
items_already_purchased = [trainset.to_raw_iid(item[0]) for item in customer_ratings]

# Get all items and remove items the customer already purchased
all_items = set(trainset.all_items())
items_to_recommend = all_items - set(trainset.to_inner_iid(item) for item in items_already_purchased)

if not items_to_recommend:
    print(json.dumps([]))  # No items to recommend
    sys.exit()

# Generate predictions
predictions = [knn_model.predict(customer_id, trainset.to_raw_iid(item)) for item in items_to_recommend]
top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]

# Add top predictions to recommendations
for prediction in top_predictions:
    product_id = int(prediction.iid)
    if product_id in input_data['product_id'].values:
        product_name = input_data.loc[input_data['product_id'] == product_id, 'product_name'].values[0]
        recommendations.append({
            'product_id': product_id,
            'product_name': product_name
        })

# Log recommendation details for debugging
# with open('C:/xampp/htdocs/FYH/fyh-source-code/cms/content/recommend_log.txt', 'a') as f:
#     f.write(f"Customer ID: {customer_id}, Recommendations: {recommendations}\n")

# Return the recommendations as JSON
print(json.dumps(recommendations))
