from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
web_search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="Search the web for up-to-date information"
)

result = web_search_tool.run("Latest news on AI")
print(result)
