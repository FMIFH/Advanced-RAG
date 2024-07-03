from langchain_openai.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class CRAG():
    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        # Prompt 
        system = """You are a grader assessing relevance of a retrieved document to a user question.
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
            
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        self.retrieval_grader = grade_prompt | structured_llm_grader