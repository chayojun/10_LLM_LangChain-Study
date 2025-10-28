from typing import List, Dict
from langchain_core.tools import BaseTool
from langchain.agents.agent_toolkits.base import BaseToolkit

# 여기에 만들 툴들을 등록
# 새로운 도구 추가
from.suggest_agent_tools import PolicySearchTool, PolicyDetailTool

class YouthpolicyrecommendationsToolkit(BaseToolkit):
    """청년 월세 관련 정책추천 툴킷입니다."""

    def get_tools(self):
        return [PolicySearchTool(), PolicyDetailTool()]