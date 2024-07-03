import os
from typing import List, Literal
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv('.env')
papers_path = os.getenv('PAPERS_PATH')

class KeywordExtract(BaseModel):
    """Route a user query to the most relevant datasource."""

    keywords: List[str] = Field(
        description="Given a user summary return a list of five keywords.",
    )
    
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class KeyWordExtractor():
    def __init__(self) -> None:
        self.keywords = {}
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        
        system = """You are a multilingual keyword extractor to be used for search engine.
        Extract the five main topics from the user given context summary of a document.
        Focus on identifying the key subjects, themes, or issues discussed in the summary. List each topic clearly and concisely in order of importance."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{context}"),
        ])
        
        # LLM
        structured_llm_extractor = llm.with_structured_output(KeywordExtract)

        # Chain
        self.keyword_chain = prompt | structured_llm_extractor 

    def extract_keywords(self) -> None:
        for paper in os.listdir(papers_path):
            if paper not in self.keywords.keys():
                summary = open(f"{papers_path}/{paper}/abstract.txt").read()
                generation = self.keyword_chain.invoke({"context": summary})
                self.keywords[paper] = generation.keywords
                
    def get_keywords_list(self):
        self.extract_keywords()
        return ", ".join([item for keyword in self.keywords.values() for item in keyword[:2]])


class AdaptiveRAG():
    def __init__(self) -> None:
        self.keyWordExtractor = KeyWordExtractor()
        # LLM with function call 
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Prompt 
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to {keywords}.
        Use the vectorstore for questions on these topics. For all else, use web-search."""

        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        
        self.question_router = route_prompt | structured_llm_router