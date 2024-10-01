import sys
import json
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

# Load the K-means model and scaler from the .pkl files
script_dir = os.path.dirname(os.path.realpath(__file__))
kmeans_model = joblib.load(os.path.join(script_dir, 'customer_segment.pkl'))
scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))

# Load the input data from a CSV file
input_data = pd.read_csv('C:/xampp/htdocs/FYH/fyh-source-code/cms/content/input_data.csv')


# Extract only the columns relevant to customer segmentation
segmentation_data = input_data[['customer_id', 'product_id', 'product_name', 'uom', 'qty', 'unit_price', 'total', 'modified']]

# Data preprocessing for segmentation
segmentation_data['modified'] = pd.to_datetime(segmentation_data['modified'])
segmentation_data = segmentation_data[segmentation_data['customer_id'].notna()]
segmentation_data.drop_duplicates(inplace=True)
segmentation_data.dropna(subset=['product_id', 'product_name', 'uom', 'qty', 'unit_price', 'total'], inplace=True)
segmentation_data = segmentation_data[segmentation_data['qty'] > 0]
segmentation_data = segmentation_data[(segmentation_data['qty'] >= 0) & (segmentation_data['unit_price'] >= 0)]

# Calculate RFM metrics
current_date = pd.Timestamp.now()

recency = segmentation_data.groupby('customer_id')['modified'].max().reset_index()
recency.columns = ['customer_id', 'last_purchase_date']
recency['recency'] = (current_date - recency['last_purchase_date']).dt.days

frequency = segmentation_data.groupby('customer_id')['modified'].count().reset_index()
frequency.columns = ['customer_id', 'frequency']

monetary = segmentation_data.groupby('customer_id')['total'].sum().reset_index()
monetary.columns = ['customer_id', 'monetary']

rfm_metrics = recency.merge(frequency, on='customer_id').merge(monetary, on='customer_id')

# Scale the RFM data
rfm_data = rfm_metrics[['recency', 'frequency', 'monetary']]
rfm_scaled = scaler.transform(rfm_data)

# Predict the clusters
rfm_metrics['Cluster'] = kmeans_model.predict(rfm_scaled)

# Manually label the clusters
cluster_analysis = rfm_metrics.groupby('Cluster').mean()

def label_clusters(row):
    if row['Cluster'] == cluster_analysis['recency'].idxmax():
        return 'Low Value'
    elif row['Cluster'] == cluster_analysis['recency'].idxmin():
        return 'High Value'
    else:
        return 'Mid Value'

rfm_metrics['Segment'] = rfm_metrics.apply(label_clusters, axis=1)

# Return the segmentation results as JSON
segmentation_results = rfm_metrics[['customer_id', 'recency', 'frequency', 'monetary', 'Segment']].to_dict(orient='records')

# Print the segmentation results as JSON
print(json.dumps(segmentation_results))
