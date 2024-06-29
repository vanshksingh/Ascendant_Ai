from langchain_community.tools import DuckDuckGoSearchRun


def search_duckduckgo(query):
    # Create an instance of the DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()

    # Run the search query
    result = search.run(query)

    # Return the result
    return result


# Example usage
if __name__ == "__main__":
    query = "Obama's first name?"
    print(search_duckduckgo(query))
