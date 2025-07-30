"""
Invesment Advisor Agent

Problem Statement:  We will carry out a project that understands investment questions written in natural language,
                    gathers reliable and up-to-date information from trusted sources, and presents it to investors in a clear,
                    concise, and current manner to support smart investment decisions.


Goals:
1. Understand the user's investment question
2. Gather reliable and up-to-date information from trusted sources
3. Present the information to the user in a clear, concise, and current manner

Technologies:
1. langchain - for building the agent, llm, tools, memory, etc.
2. openai - for generating the response, gpt-4.1-nano
3. CoinGecko - currency conversion, free
4. Finnhub - for getting the stock data, free api
5. DuckDuckGo - for searching the web, no api needed


Plan
1. Tools, Duckduckgo search, CoinGecko conversion, Finnhub stock data
2. llm 
3. agent
4. answer questions with agent




Install libraries, freeze


Suggestions for the future:
1. Add memory to the agent
2. Add tools, better search, better summarization, better analysis
3. Add a UI, chat interface with streamlit/fastapi
4. plan and execute a strategy to make money

"""

from langchain.agents import initialize_agent, AgentType # @start langchain agent and agent type
from langchain.chat_models import ChatOpenAI # @start langchain chat model
from tools.search_tool import search # our search tool
from tools.currency_converter import convert_usd_to_try # our currency converter tool
from tools.market_api import get_stock_data # our market api tool

from dotenv import load_dotenv
import os

from langchain.prompts import PromptTemplate # personalized prompt template
from langchain.chains import LLMChain # chain of llm = model + prompt template

# .env file load api keys
load_dotenv()

# openai chat model
llm = ChatOpenAI(
    model_name="gpt-4.1-nano", # llm model
    temperature=0, # temperature
    openai_api_key=os.getenv("OPENAI_API_KEY") # openai api key
    )

# tools
tools = [search, convert_usd_to_try, get_stock_data]

# user question {input} in prompt with placeholder 
investment_agent = PromptTemplate.from_template(
        """
        You are an expert financial investment advisor. 
        User's question: {input}
        """
    )

# chain of llm = model + prompt template

llm_chain = LLMChain(llm = llm, prompt = investment_agent)

# start langchain agent ( define ai agent )

agent = initialize_agent(
    tools = tools, # tools
    llm = llm, # llm
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # agent type
    verbose=True, # write the agent's actions to the console
    handle_parsing_errors=True # handle parsing errors
)

if __name__ == "__main__":
    print("Welcome to the investment agent!, type 'exit' to end the program")

    # infinite loop
    while True:
        # get user input
        query = input("Enter your question: ")

        if query.lower() == "exit":
            print("Exiting the program...")
            break

        try:
            # run the agent
            response = agent.invoke({"input": query})

            print("Agent's response: ", response)

        except Exception as e:
            print(f"Error: {e}")