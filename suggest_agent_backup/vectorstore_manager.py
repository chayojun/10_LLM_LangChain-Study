import os
import json
import re
from unidecode import unidecode
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

class VectorStoreManager:
    def __init__(self, header_db_path: str = "source/vectorstore/header_db", body_db_path: str = "source/vectorstore/body_db"):
        self.header_db_path = header_db_path
        self.body_db_path = body_db_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

        self.header_vectorstore = Chroma(
            persist_directory=self.header_db_path,
            embedding_function=self.embeddings,
            collection_name="header_db"
        )

    def _generate_summary(self, documents: list[Document]) -> str:
        prompt = ChatPromptTemplate.from_template(
            """
            다음은 정책 문서의 일부 내용입니다. 내용을 분석하여 아래 JSON 형식에 맞게 핵심 정보를 요약해 주세요.
            - "policy_category": 정책의 대분류 (예: 주거, 금융, 교육)
            - "policy_name": 정책의 상세 이름 또는 제목 (예: 월세 지원, 학자금 대출)
            - "policy_summary": 정책을 간략하게 요약하고 전문적 단어를 쉽게 풀어 작성 (예: 임차보증금, 중개보수) 
            - "policy_target": 정책의 주요 대상 (예: 청년, 신혼부부, 자영업자)
            - "policy_benefit": 정책의 핵심 혜택 (예: 월 20만원 지원, 최대 240만원)

            내용:
            {context}

            JSON 출력:
            """
        )
        
        output_parser = StrOutputParser()
        chain = prompt | self.llm | output_parser
        
        full_content = "\n".join([doc.page_content for doc in documents])
        
        summary_json_str = chain.invoke({"context": full_content})
        return summary_json_str

    def process_and_store_document(self, file_path: str):
        # 1. 문서 로드
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 2. 정책 본문(body) 벡터 저장소 생성 및 저장
        policy_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # ChromaDB 컬렉션 이름 규칙에 맞게 변환
        # 한글을 ASCII로 음차 변환
        # 예: '청년임차보증금 이자지원정책' -> 'cheongnyeonimchabojeunggeum ijajiwonjeongchaek'
        # 이 부분은 한국어로 들어가면 에러가 나서 이렇게 작성해놨습니다.
        ascii_name = unidecode(policy_name)
        # 영문, 숫자, ., _, - 만 허용
        safe_policy_name = re.sub(r'[^a-zA-Z0-9._-]', '', ascii_name)
        # 시작과 끝이 영문/숫자가 아니면 제거
        safe_policy_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', safe_policy_name)
        # 최소 길이 보장
        if len(safe_policy_name) < 3:
            safe_policy_name = f"policy-{safe_policy_name}"
        # 최대 길이 제한
        safe_policy_name = safe_policy_name[:60] # ChromaDB 최대 길이에 맞춰 조정

        original_collection_name = f"{safe_policy_name}_body"
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        body_vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.body_db_path,
            collection_name=original_collection_name
        )
        body_vectorstore.persist()
        print(f"'{policy_name}' 본문을 '{original_collection_name}' 컬렉션에 저장했습니다.")

        # 3. 정책 요약(header) 생성
        summary_json_str = self._generate_summary(documents)
        
        # LLM 응답에서 JSON 코드 블록 정리
        # 이 부분은 가끔 llm 이 ```json ... ``` 형태로 감싸서 보내주는 경우가 있어서 처리하는 부분입니다.
        if "```json" in summary_json_str:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", summary_json_str)
            if match:
                summary_json_str = match.group(1)

        try:
            summary_data = json.loads(summary_json_str)
            print(f"생성된 요약: {summary_data}")
        except json.JSONDecodeError:
            print(f"오류: LLM이 유효한 JSON을 생성하지 못했습니다. 받은 내용: {summary_json_str}")
            return

        # 4. 헤더 정보 구성 및 벡터 저장소에 저장
        header_content = {
            "summary": summary_data,
            "policy_name": policy_name,
            "original_collection_name": original_collection_name
        }
        
        header_document = Document(
            page_content=json.dumps(header_content, ensure_ascii=False),
            metadata={
                "policy_name": policy_name,
                "source": file_path
            }
        )
        
        self.header_vectorstore.add_documents([header_document])
        self.header_vectorstore.persist()
        print(f"'{policy_name}' 헤더를 'header_db' 컬렉션에 저장했습니다.")


if __name__ == '__main__':
    # 여기 Root 폴더 기준으로 넣어야 됩니다.
    # 예시 : 'source/original_docs/sample_policy.pdf'
    sample_pdf_path = "source\\original_docs\\서울시 청년월세지원정책.pdf"

    if not os.path.exists(sample_pdf_path):
        print(f"'{sample_pdf_path}' 파일을 찾을 수 없습니다. 테스트를 위해 해당 경로에 PDF 파일을 위치시켜 주세요.")
    else:
        manager = VectorStoreManager()
        manager.process_and_store_document(sample_pdf_path)
        print("\n작업 완료.")

        # 실제로 헤더 검색을 해서 제대로 저장됐는지 확인하는 부분입니다.
        # 여기는 필수 아닌데, 아래에 get_relevant_documents() 부분에 원하는 키워드나 문구로 바꿔서 테스트해보시면 됩니다.
        retriever = manager.header_vectorstore.as_retriever(search_kwargs={"k": 1})
        retrieved_docs = retriever.get_relevant_documents("청년임차보증금")
        if retrieved_docs:
            print("\n[검색 테스트]")
            retrieved_content = json.loads(retrieved_docs[0].page_content)
            print(f"검색된 정책: {retrieved_content['policy_name']}")
            print(f"요약: {retrieved_content['summary']}")
            print(f"원본 컬렉션: {retrieved_content['original_collection_name']}")
        else:
            print("\n[검색 테스트] 관련된 정책을 찾지 못했습니다.")