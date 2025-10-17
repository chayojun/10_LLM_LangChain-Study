from typing import List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# 스케줄 등록하는 리스트
# 1. 스케줄 등록 도구
# 2. 스케줄 확인 도구
# 3. 스케줄 삭제 도구
# 4. 스케줄 완료 도구

dol_schedule : List[str] = []

# 1-1 스케줄 스키마 설정
class AddToDoInput(BaseModel):
    item : str = Field(description="오늘 할 돌머리 스케줄 항목")

# 1-2 스케줄 등록 도구 설정
class AddToDoTool(BaseTool):
    name : str = "add_todo"
    description : str = "돌마리 스케줄에 새 항목을 추가합니다."
    args_schema : Type[BaseModel] = AddToDoInput

    def _run(self, item: str) -> str:
        dol_schedule.append(item)
        return f"{item}이 돌마리 할일 스케줄에 등록 되었습니다"
    
# 2-1 스케줄 확인 스키마

# 2-2 스케줄 확인 도구 설정
class ViewToDoTool(BaseTool):
    name : str = "view_todos"
    description : str = "현재 돌마리 스케줄 전체 목록을 보여줍니다." 

    def _run(self):
        
        if not dol_schedule:
            return "할일이 없어요"
        all_schedule = "\n".join(dol_schedule)
        return f"할일 목록은 : {all_schedule}\n입니다."
    
# 3-1 스케줄 삭제 스키마
class DeleteToDoTool(BaseModel):
    item : str = Field(description = "삭제할 항목의 이름")


# 3-2 스케줄 삭제 삭제 설정
class DeleteToDoTool(BaseTool):
    name : str = "delete_todo"
    description : str = "돌마리 스케줄 항목을 삭제 합니다."
    args_schema : Type[BaseModel] = DeleteToDoTool

    def _run(self, item: str) -> str:
         
        try:
            dol_schedule.remove(item)
            return f"{item}이 돌마리 할일 스케줄에서 삭제 되었습니다."
        except ValueError:
            return f"{item}이 돌마리 할일 스케줄에 없습니다."
        

# # 4. 스케줄 완료 설정
# class CompleteToDoTool(BaseTool):
#     name : str = "complete_todo"
#     description : ""  복습 시간에 완성해보기