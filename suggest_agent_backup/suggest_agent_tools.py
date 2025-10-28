from typing import Optional, Dict, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
import os

# 환경 변수 로드
load_dotenv()

class PolicySearchTool:
    """정책 검색 도구"""
    
    def __init__(self, header_db_path: str = "source/vectorstore/header_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.header_vectorstore = Chroma(
            persist_directory=header_db_path,
            embedding_function=self.embeddings,
            collection_name="header_db"
        )
    
    def run(self, user_query: str) -> Dict:
        """
        사용자 질문을 바탕으로 가장 적합한 정책을 찾습니다.
        """
        try:
            # 헤더 검색
            retriever = self.header_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1}
            )
            headers = retriever.get_relevant_documents(user_query)
            
            if not headers:
                return {
                    "status": "error",
                    "message": "관련 정책을 찾을 수 없습니다."
                }
            
            header_content = json.loads(headers[0].page_content)
            return {
                "status": "success",
                "policy_name": header_content.get("policy_name"),
                "collection_name": header_content.get("original_collection_name"),
                "summary": header_content.get("summary", {})
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"정책 검색 중 오류 발생: {str(e)}"
            }

class PolicyDetailTool:
    """정책 상세 정보 추출 도구"""
    
    def __init__(self, body_db_path: str = "source/vectorstore/body_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model_name="gpt-4-mini", temperature=0)
        self.body_db_path = body_db_path
        
    def run(self, collection_name: str) -> Dict:
        """
        지정된 정책의 상세 정보를 추출하여 JSON 형식으로 반환합니다.
        """
        try:
            # 본문 벡터 저장소 로드
            vectorstore = Chroma(
                persist_directory=self.body_db_path,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            # 검색 키워드
            detail_queries = [
                "필수 조건", "우대 조건", "혜택", 
                "필수 서류", "정책 지역", "제외 대상자"
            ]
            
            retrieved_chunks = []
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.01}
            )
            
            # 각 키워드로 검색
            for query in detail_queries:
                docs = retriever.get_relevant_documents(query)
                retrieved_chunks.extend(docs)
            
            # LLM으로 JSON 형식 답변 생성
            template = """
            다음 정책 정보를 분석하여 정확히 아래 JSON 형식으로 출력하세요:
            {
                "eligibility_criteria": string,  // 필수 조건
                "priority_subjects": string,     // 우대 조건
                "benefits": string,             // 혜택
                "required_documents": [string],  // 필수 서류 목록
                "policy_region": string,        // 정책 지역
                "exclusions": string           // 제외 대상자
            }
            
            정보가 없는 항목은 "정보 없음"으로 표시하세요.
            JSON 형식만 출력하고 다른 설명은 하지 마세요.

            정책 정보:
            {context}
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
            
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"context": context})
            
            # JSON 파싱
            detail_json = json.loads(result)
            return {
                "status": "success",
                "details": detail_json
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"상세 정보 추출 중 오류 발생: {str(e)}"
            }

# # 사용 예시
# if __name__ == "__main__":
#     # 1. 정책 검색
#     search_tool = PolicySearchTool()
#     query = "서울시에 사는 27세 청년입니다. 월세 지원 정책 찾아주세요."
#     policy = search_tool.run(query)
    
#     if policy["status"] == "success":
#         print(f"검색된 정책: {policy['policy_name']}")
        
#         # 2. 상세 정보 추출
#         detail_tool = PolicyDetailTool()
#         details = detail_tool.run(policy["collection_name"])
        
#         if details["status"] == "success":
#             print("\n정책 상세 정보:")
#             print(json.dumps(details["details"], ensure_ascii=False, indent=2))