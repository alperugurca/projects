import requests # http client

# api adress

API_URL = "http://127.0.0.1:8000/chat" # fastapi server adress and endpoint

# take user data

name = input("Enter your name: ")
age = int(input("Enter your age: "))

print("\n Chat started. For exit, type 'exit'")

# create a loop take message and send to server

while True:

    user_msg = input(f"{name}: ")
    if user_msg.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    # send message to server
    payload = {"name": name, "age": age, "message": user_msg}

    try:
        res = requests.post(API_URL, json=payload, timeout=30)

        if res.status_code == 200:
            print(f"Assistant: {res.json()['response']}")
        else:
            print("Error", res.status_code, res.text)
    except requests.exceptions.RequestException as e:
        print("Connection error", e)