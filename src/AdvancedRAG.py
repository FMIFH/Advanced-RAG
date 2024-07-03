
from typing import List, TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

from langgraph.graph import END, StateGraph, Graph
from src.AdaptiveRAG import AdaptiveRAG

from src.CorrectiveRAG import CRAG
from src.SelfRAG import HallucinationChecker, ResponseChecker


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    web_search : bool
    grounded : bool
    documents : List[str]
    steps : int
    end : bool

class RAG():
    def __init__(self) -> None:
        # LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = hub.pull("rlm/rag-prompt")
        # Chain
        self.rag_chain = prompt | llm | StrOutputParser()


class AdvancedRAG():
    def __init__(self, vs:VectorStore ) -> None:
        self.retriever = vs.as_retriever()
        self.rag = RAG()
        self.crag = CRAG()
        self.arag = AdaptiveRAG()
        self.hallucinationChecker = HallucinationChecker()
        self.responseChecker = ResponseChecker()
        self.web_search_tool = TavilySearchResults(k=3)
        self.app = self.make_app()
    
    def retrieve(self,state: GraphState) -> GraphState:
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print('-- RETRIEVE --')
        question = state['question']
        docs = self.retriever.invoke(question)

        return {
            'documents' : docs,
            'question' : question
        }
        
    def generate(self, state: GraphState) -> GraphState:
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state['question']
        docs = state['documents']
        generation = self.rag.rag_chain.invoke({"context": docs, "question": question})

        return {
            'documents' : docs,
            'question' : question,
            'generation': generation
        }
        
    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state['question']
        docs = state['documents']
        new_docs = []
        web_search = False
        for doc in docs:
            res = self.crag.retrieval_grader.invoke({"question": question, "document": doc})
            score = res.binary_score
            
            if score.lower() == 'yes':
                print("---GRADE: DOCUMENT RELEVANT---")
                new_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = True
        
        return {
            'documents' : new_docs,
            'question' : question,
            'web_search' : web_search
        }

    def web_search(self, state: GraphState) -> GraphState:
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {
            "documents": documents,
            "question": question
        }

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]

        if web_search:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
        
    def grade_generation_v_documents(self, state: GraphState) -> GraphState:
        """
        Determines whether the generation is grounded in the document and answers question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended hallucination addressing measure
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        docs = state['documents']
        generation = state["generation"]
        steps = state["steps"]
        score = self.hallucinationChecker.hallucination_grader.invoke({"documents": docs, "generation": generation})
        grade = score.binary_score
        
        grounded = grade.lower() == 'yes'
        if grounded:
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        else:
            print("---DECISION: REGENERATE---")
            if steps is not None:
                steps += 1
            else:
                steps = 1
            
            
        
        return {
            "question" : question,
            "generation" : generation,
            "grounded" : grounded,
            "steps" : steps,
            "documents" : docs
        }
        
    def grade_generation_v_question(self, state: GraphState) -> GraphState:
        """
        Determines whether the generation is grounded in the document and answers question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended question addressing measure
        """

        print("---CHECK GENERATON ADDRESSES QUESTION---")
        question = state["question"]
        docs = state['documents']
        generation = state["generation"]
        steps = state["steps"]


        score = self.responseChecker.answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score

        end = False
        if grade.lower() == 'yes':
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            end = True
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            if steps is not None:
                steps += 1
            else:
                steps = 1
            
        
        return {
            "question" : question,
            "generation" : generation,
            "end" : end,
            "documents" : docs
        }
        
    def hallucinated(self, state: GraphState) -> str:
        """
        Determines whether to regenerate an answer based on halluciation

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        
        grounded= state["grounded"]
        
        if grounded:
            return "no"
        else:
            steps = state["steps"]
            if steps < 5:
                return "regenerate"
            else:
                print("---EXCEEDED STEPS---")
            return "end"
        
    def answers_question(self, state: GraphState) -> str:
        """
        Determines whether to regenerate an answer based on the quality of the answer

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        
        end = state["end"]
        
        if end:
            return "yes"
        else:
            steps = state["steps"]

            if steps < 5:
                return "regenerate"
            else:
                print("---EXCEEDED STEPS---")
            return "yes"
        
        
    def route_question(self, state: GraphState) -> str:
        """
        Route question to web search or RAG 

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.arag.question_router.invoke({'keywords':self.arag.keyWordExtractor.get_keywords_list(), "question": question})   
        if source.datasource == 'websearch':
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source.datasource == 'vectorstore':
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
       
    def make_app(self) -> Graph: 
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("websearch", self.web_search)  # web search
        workflow.add_node("check_hallucination", self.grade_generation_v_documents) #hallucination check
        workflow.add_node("check_answer", self.grade_generation_v_question) #hallucination check


        # Build graph
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_edge("generate", "check_hallucination")
        workflow.add_conditional_edges(
            "check_hallucination",
            self.hallucinated,
            {
                "regenerate": "generate",
                "end": END,
                "no": "check_answer",
            },
        )
        workflow.add_conditional_edges(
            "check_answer",
            self.answers_question,
            {
                "regenerate": "websearch",
                "yes": END,
            },
        )

        # Compile
        app = workflow.compile()
        return app