import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_math.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(f"Available tools: {[tool.name for tool in tools]}")
            model = ChatOpenAI(model="gpt-4o")
            agent = create_react_agent(model, [])
            agent.tools.extend(tools)
            agent_response = await agent.ainvoke({"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]})
            final_message = agent_response['messages'][-1]
            print(final_message.content)

if __name__ == "__main__":
    asyncio.run(main())
