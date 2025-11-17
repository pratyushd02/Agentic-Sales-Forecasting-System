# test_simple.py
import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test():
    server_params = StdioServerParameters(command=sys.executable, args=["mcp_sales_server.py"])
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✓ Session initialized")
                return "Success"
    except BaseException as e:
        print(f"Error type: {type(e)}")
        print(f"Error: {e}")
        return "Failed"

result = asyncio.run(test())
print(f"Result: {result}")