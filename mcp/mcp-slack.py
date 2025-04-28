import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    # Set environment variables for the Slack MCP server
    env = {
        "SLACK_BOT_TOKEN": "xoxb-",
        "SLACK_TEAM_ID": "",
        "SLACK_CHANNEL_IDS": ""
    }
    
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-slack"],
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(f"Available tools: {[tool.name for tool in tools]}")
            
            # Create the agent with tools first
            llm = ChatOpenAI(model="gpt-4")
            agent = create_react_agent(llm, tools)  # Pass tools during creation
            
            # Invoke the agent
            agent_response = await agent.ainvoke({
                "messages": [{
                    "role": "user", 
                    "content": "List the available channels in my Slack workspace"
                }]
            })
            final_message = agent_response['messages'][-1]
            print(final_message.content)

if __name__ == "__main__":
    asyncio.run(main())
