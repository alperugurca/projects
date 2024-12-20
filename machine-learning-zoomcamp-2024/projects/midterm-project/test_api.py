import requests

# Test data
test_data = {
    'Hours per day': 3.0,
    'Anxiety': 7.0,
    'Depression': 5.0,
    'Insomnia': 6.0,
    'OCD': 4.0,
    'Fav genre': 'Rock'
}

# Make prediction request
response = requests.post('http://localhost:5000/predict', json=test_data)

# Print status code to check if the request was successful
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")  # Print raw response for debugging

if response.status_code == 200:
    try:
        print(response.json())  # Try parsing JSON if response is successful
    except requests.exceptions.JSONDecodeError:
        print("Error decoding JSON response")
else:
    print(f"Failed to get a valid response. Status code: {response.status_code}")
