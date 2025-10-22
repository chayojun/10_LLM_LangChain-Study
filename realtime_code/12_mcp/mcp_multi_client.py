import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from urllib.parse import urlencode
SMITHERY_AI_KEY  = os.getenv("SMITHERY_AI_KEY")
SMITHERY_AI_PROFILE = os.getenv("SMITHERY_AI_PROFILE")

base_url = "https://server.smithery.ai/@isdaniel/mcp_weather_server/mcp"
params = urlencode({"api_key": SMITHERY_AI_KEY, "profile": SMITHERY_AI_PROFILE})
url = f"{base_url}?{params}"

load_dotenv()

# 실제 실행 파트
async def main():
    print("멀티 클라이언트 세팅중...")

    client = MultiServerMCPClient(
        {
            "Math" : {
                "command" : "python",
                "args" : [r"C:\POTENUP\10_LLM_LangChain-Study\realtime_code\12_mcp\odd_math_server.py"],
                "transport" : "stdio"
            },
            # "Weather" : {
            #     "url" : "http://localhost:8100/mcp",
            #     "transport" : "streamable_http"
            # }
              "smithery_Weather" : {
                "url" : url,
                "transport" : "streamable_http"
            },
}

)

    tools = await client.get_tools()
    for item in tools:
        print("가져온 도구는 : ", item.name)

    # agent 만들기
    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

############################################################################################
    # 더하기 도구 체크
    agent_excutor = create_react_agent(model, tools)
    response = await agent_excutor.ainvoke(
        {"messages" : [HumanMessage(content="1+2를 이상하게 계산하면?")]}
    )
############################################################################################    
    print(response["messages"][-1].content)

    # 곱하기 도구 체크
    response = await agent_excutor.ainvoke(
        {"messages" : [HumanMessage(content="2 * 3을 이상하게 계산하면?")]}
    )
############################################################################################
    print(response["messages"][-1].content)

    # 날씨 도구 체크
    response = await agent_excutor.ainvoke(
    {"messages" : [HumanMessage(content="석촌역의 날씨는?")]}
    )
    print(response["messages"][-1].content)
############################################################################################
if __name__ == "__main__":
    asyncio.run(main())