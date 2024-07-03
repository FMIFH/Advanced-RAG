from langchain_openai.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
    

class HallucinationChecker():
    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)
        # Prompt 
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        self.hallucination_grader = hallucination_prompt | structured_llm_grader
        


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

class ResponseChecker():
    def __init__(self) -> None:
        # LLM with function call 
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        # Prompt 
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        self.answer_grader = answer_prompt | structured_llm_grader