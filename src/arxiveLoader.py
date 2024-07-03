import arxiv
from typing import List
from dotenv import load_dotenv
import os

load_dotenv('.env')
papers_path = os.getenv('PAPERS_PATH')

class ArxiveLoader():
    def __init__(self) -> None:
        if papers_path is None:
            raise ValueError('STORAGE_PATH environment variable is not set')
        os.makedirs(papers_path, exist_ok=True)
        self.client = arxiv.Client()
        self.papers = []
        
    def download_papers(self, ids:List[str]) -> List[str]:
        new_papers_dir = []
        papers = self.client.results(arxiv.Search(id_list=ids))
        for paper in papers:
            directory_path = f"{papers_path}/{paper.title}"
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                print(f"The file '{paper.title}' already exists.")
                continue
            print(paper, paper.title)
            os.makedirs(directory_path)
            paper.download_pdf(dirpath=directory_path, filename=f"{paper.title}.pdf")
            with open(f"{directory_path}/abstract.txt", "x") as file:
                file.write(paper.summary)
            new_papers_dir.append(f"{directory_path}/{paper.title}.pdf")
        return new_papers_dir
