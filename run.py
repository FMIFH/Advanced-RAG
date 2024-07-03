from src.AdvancedRAG import AdvancedRAG
from src.arxiveLoader import ArxiveLoader
from src.vectorDB import vectorStore
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeColors

from src.arxiveLoader import ArxiveLoader

PAPERS = ['2205.14135', '2005.14165', '1706.03762', '2403.14403', '2310.11511', '2401.15884']


if __name__ == "__main__":
    loader = ArxiveLoader()
    new_papers = loader.download_papers(PAPERS)
    chroma = vectorStore()
    chroma.add_documents(new_papers)
    advancedRAG= AdvancedRAG(chroma)
    app = advancedRAG.make_app()
    inputs = {"question": "what is flash attention?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")
    print(value["generation"])