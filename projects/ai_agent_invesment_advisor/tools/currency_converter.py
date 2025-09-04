"""
usd to try
"""

from langchain.tools import tool # @langchain agent tool decorator
import requests

@tool # @ langchain tool decorator
def convert_usd_to_try(amount: float) -> str:
    """
    Convert USD to TRY.
    Args:
        amount: The amount in USD to convert to TRY (must be a number)
    Returns:
        A string with the conversion result
    """
    try:
        # Convert input to float if it's a string or dict
        if isinstance(amount, dict) and 'action_input' in amount:
            amount = float(amount['action_input'])
        elif isinstance(amount, str):
            amount = float("".join(filter(lambda c: c.isdigit() or c == ".", amount)))
        elif not isinstance(amount, (int, float)):
            raise ValueError("Amount must be a number")

        #coingecko api usd/try convert rate
        url = "https://api.coingecko.com/api/v3/simple/price?ids=usd&vs_currencies=try" # generalization like {ids}

        # get request
        response = requests.get(url)

        # if response is not 200, raise an error

        if response.status_code != 200:
            return f"Error: {response.status_code}"
        
        # take json data as dict
        data = response.json()

        # take rate value
        rate = data["usd"]["try"]

        # user amount * rate
        result = amount*rate

        # 
        return f"{amount} USD = {result:.2f} TRY (TRY rate: {rate:.2f})"
    except Exception as e:
        return f"Error: {e}"
    
if __name__ == "__main__":
    #test
    test_amount = 100
    print(f"Testing with amount: {test_amount} USD > TRY")
    print(convert_usd_to_try.run({"amount": test_amount}))