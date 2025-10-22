from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Weather",
    host="0.0.0.0",
    port=8100
    )

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location"""
    return "석촌역의 날씨에 대해서 알려줘"

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http"
    )