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
    
    def __init__(self, header_db_path: str = "../../source/vectorstore/header_db"):
        self.header_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.header_vectorstore = Chroma(
            persist_directory=header_db_path,
            embedding_function=self.header_embeddings,
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
                "summary": header_content.get("policy_summary")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"정책 검색 중 오류 발생: {str(e)}"
            }



class PolicyDetailTool:
    """정책 상세 정보 추출 도구"""
    
    def __init__(self, body_db_path: str = "../../source/vectorstore/body_db"):
        self.body_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.body_db_path = body_db_path
        
    def run(self, collection_name: str) -> Dict:
        """
        지정된 정책의 상세 정보를 추출하여 JSON 형식으로 반환합니다.
        """
        try:
            print(f"Collection 로드 중: {collection_name}")
            
            # 본문 벡터 저장소 로드
            vectorstore = Chroma(
                persist_directory=self.body_db_path,
                embedding_function=self.body_embeddings,
                collection_name=collection_name
            )
            
            # 섹션별 상세 검색 쿼리
            detail_queries = [
                "이 정책에 신청하기 위한 필수 자격 조건은 무엇인가요?",
                "이 정책에서 우대를 받을 수 있는 조건이나 가산점 항목, 추가 제출이 필요한 서류는 무엇인가요?",
                "이 정책을 통해 받을 수 있는 구체적인 혜택이나 지원 내용은 무엇인가요?",
                "이 정책에 신청하기 위해 반드시 제출해야 하는 서류는 무엇인가요?",
                "이 정책이 적용되는 대상 지역이나 거주지 요건은 어디인가요?",
                "이 정책의 지원 대상에서 제외되는 사람은 누구인가요?"
            ]
            
            
            # 섹션별 문서 수집
            all_chunks = []
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.05
                }
            )
            
            print("문서 검색 중...")
            for queries in detail_queries:
                docs = retriever.get_relevant_documents(queries)
                all_chunks.extend(docs)
            
            # 중복 제거 (더 엄격한 기준)
            unique_chunks = []
            seen_contents = set()
            for chunk in all_chunks:
                content = chunk.page_content.strip()
                content_hash = hash(content[:300])
                if content_hash not in seen_contents:
                    unique_chunks.append(chunk)
                    seen_contents.add(content_hash)
            
            print(f"검색된 고유 문서 수: {len(unique_chunks)}")
            
            # 컨텍스트 준비 (충분한 정보 제공)
            context = "\n\n--- 문서 구분 ---\n\n".join([
                chunk.page_content for chunk in unique_chunks[:20]
            ])
            
            print(f"컨텍스트 길이: {len(context)} 자\n")
            
            # 강화된 프롬프트
            template = """당신은 서울시 청년 주거지원 정책 문서 분석 전문가입니다.

            제공된 정책 문서를 꼼꼼히 읽고 아래 JSON 형식으로 정확하게 추출하세요.

            {{
                "eligibility_criteria": [string],  // 필수 조건
                "priority_subjects": [string],     // 우대 조건 = [우대 조건, 추가 서류]
                "benefits": [string],             // 혜택
                "required_documents": [string],  // 필수 서류 목록
                "policy_region": [string],        // 정책 지역
                "exclusions": [string]           // 제외 대상자
            }}

            필드별 작성 지침:

            1. eligibility_criteria (필수 자격요건):
            - 연령: 정확한 나이 범위와 출생연도 (예: 만 19~39세, 1985.1.1~2006.12.31)
            - 거주: 주민등록, 전입신고 관련 모든 조건
            - 가구: 세대 구성 요건 (예: 1인 가구, 세대주)
            - 소득: 기준중위소득 비율, 건강보험료 기준
            - 주거: 보증금/월세 한도, 주거 형태
            - 문서에서 찾은 모든 필수 조건을 빠짐없이 포함
            
            2. priority_subjects (우대 조건):
            - 문서에 우대/우선 선발 조건이 있으면 [["조건", "추가서류"]] 형태로
            - 없으면 빈 리스트 []
            
            3. benefits (혜택):
            - 지원 금액: 정확한 금액 (예: 월 최대 20만원)
            - 지원 기간: 개월 수 (예: 12개월)
            - 지급 방식: 지급 주기와 방법 (예: 격월 25일 계좌 입금)
            - 총 지원액: 생애 최대 지원 금액 (예: 최대 240만원)
            - 문서에 명시된 모든 혜택 정보 포함
            
            4. required_documents (필수 서류):
            - 모든 신청자가 반드시 제출해야 하는 기본 서류
            - 각 서류명을 정확하게 (예: "확정일자가 날인된 임대차계약서 전체 사본 1부")
            - 문서에서 찾은 모든 필수 서류를 빠짐없이 리스트로
            
            5. policy_region (정책 지역):
            - "서울시" 또는 "서울특별시"
            
            6. exclusions (제외 대상자):
            - 신청할 수 없는 모든 경우를 나열
            - 중복 수혜, 기수혜자, 주택 소유, 공공임대 거주 등
            - 문서에서 찾은 모든 제외 대상을 상세히 포함

            중요 규칙:
            - 문서에 명시된 내용만 사용 (추측 금지)
            - 모든 수치는 정확하게 (금액, 개월, 비율 등)
            - 정보가 정말 없는 경우에만 "정보 없음"
            - 문서 내용을 요약하지 말고 상세히 작성
            - JSON 형식만 출력 (마크다운 코드 블록 사용 금지)

            정책 문서:
            {context}
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            print("LLM 분석 중...")
            result = chain.invoke({"context": context})
            
            # JSON 추출
            result = result.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            # JSON 파싱
            detail_json = json.loads(result)
            
            print("추출 완료\n")
            
            return {
                "status": "success",
                "details": detail_json
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {str(e)}")
            print(f"LLM 응답:\n{result[:500]}")
            return {
                "status": "error",
                "message": f"JSON 파싱 실패: {str(e)}",
                "raw_response": result if 'result' in locals() else "응답 없음"
            }
        except Exception as e:
            import traceback
            print(f"오류 발생: {str(e)}")
            traceback.print_exc()
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