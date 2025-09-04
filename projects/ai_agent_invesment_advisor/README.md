# AI Investment Advisor

An intelligent agent that helps users make informed investment decisions by understanding natural language questions, gathering real-time market data, and providing clear, actionable insights.

## Features

- Natural language understanding for investment queries
- Real-time market data integration
- Currency conversion capabilities
- Web search for latest financial information
- Interactive command-line interface

## Technologies

- LangChain for agent orchestration
- OpenAI GPT-4 for natural language processing
- CoinGecko API for currency conversion
- Finnhub API for stock market data
- DuckDuckGo for web searches

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ```
4. Run the agent:
   ```bash
   python agent_main.py
   ```

## Usage

Simply run the program and ask investment-related questions in natural language. Type 'exit' to end the program.

Example questions:
- "What is the current price of Tesla stock?"
- "Convert 100 USD to Turkish Lira"
- "What are the latest news about Bitcoin?"

## Future Improvements

- Memory integration for contextual conversations
- Enhanced analysis tools
- Web interface using Streamlit/FastAPI
- Automated investment strategy execution 