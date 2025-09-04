import os
import requests
from dotenv import load_dotenv # .env file to store api key

# .env import

load_dotenv()

# .env GEMINI_API_KEY
api_key = os.getenv("GEMINI_API_KEY")

# if api_key is not found, raise an error

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

# gemini 2.0 flash api url
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# api call http

headers = {
    "Content-Type": "application/json", # Json format data
    "X-Goog-API-Key": api_key # api key for authentication
}

def get_gemini_response(prompt: str) -> str: # gemini api send prompt and get response
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt # user prompt
                    }
                ]
            }
        ]
    }

    # send request to gemini api
    response = requests.post(url, headers=headers, json=payload)

    # if http status code is 200, return the response
    if response.status_code == 200:
        try:
            result = response.json() # json format to python dict
            return result["candidates"][0]["content"]["parts"][0]["text"] # return the response
        except Exception as e:
            #if json is not valid, return the error
            return f"Error: {e}"
    else:
        return f"Error: {response.status_code}: {response.text}"
    
# detect intent from message function
def detect_intent(message):
    # for gemini
    prompt = f"""
                Classify the user's sentence below.

                Return only one label
                - notes_summary (if user want to see notes summary)
                - events_summary (if user want to see events summary)
                - normal_chat (if user want to chat with you)

                Sentence: "{message}"
                Return only one label (example: notes_summary)
            """
    # send prompt to gemini and get response
    response = get_gemini_response(prompt)
    return response.strip().lower()

if __name__ == "__main__":
    user_input = input("Enter your prompt: ")
    response = get_gemini_response(user_input)
    print(f"Smart assistant: {response}")