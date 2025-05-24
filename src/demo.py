import requests
import json

# For inference
response = requests.post(
    'http://localhost:8888/infer',
    headers={'Content-Type': 'application/json'},
    json={'prompt': 'You are a helpful assistant. \nQuestion:\nWhat is 2+2?.\nAnswer:\n'}
)
print(response.json())

# For ad insertion
response = requests.post(
    'http://localhost:8888/insert_native_ads',
    headers={'Content-Type': 'application/json'},
    json={'text': 'This is some sample text content.'}
)
print(response.json())