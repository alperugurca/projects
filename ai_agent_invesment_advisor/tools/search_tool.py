"""

duckduckgosearchrun in langchain
for web search

"""

from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

if __name__ == "__main__": # only to test the tool
    query = "what is the dollar to turkish lira exchange rate"

    # sent to search engine
    result = search.run(query)

    # print results
    print(f"Results: \n{result}")