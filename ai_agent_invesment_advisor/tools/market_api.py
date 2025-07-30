"""
Finnhub API is a tool that allows you to get the stock data of a company.
api_key from finnhub.io
"""
from langchain.tools import tool # @langchain agent tool decorator
import requests
import os
from dotenv import load_dotenv

load_dotenv()

@tool # @ langchain tool decorator
def get_stock_data(ticker: str) -> str:
    """
        Get the stock data of a company.
        ticker is the symbol of the company.
    """

    try:
        # take finnhub api key from .env
        api_key = os.getenv("FINNHUB_API_KEY")

        # if api_key is not found, raise an error
        if not api_key:
            raise ValueError("FINNHUB_API_KEY is not found in .env file")
        
        # finnhub api url
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"

        # get request
        response = requests.get(url)

        # if response is not 200, raise an error
        if response.status_code != 200:
            return f"API Error: {response.status_code}"
        
        # Decode json data
        data = response.json()

        current = data.get("c") # current price
        open_price = data.get("o") # open price
        high_price = data.get("h") # high price
        low_price = data.get("l") # low price
        previous_close = data.get("pc") # previous close price

        # return stock data
        return (
            f"{ticker} Current price: {current} USD\n"
            f"Open price: {open_price} USD\n"
            f"High price: {high_price} USD\n"
            f"Low price: {low_price} USD\n"
            f"Previous close: {previous_close} USD\n"
        )
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print(get_stock_data.run({"ticker": "GOOGL"}))
