import os
from dotenv import load_dotenv
import fitz  
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class RAGSystem:
    def __init__(self, pdf_path):
        load_dotenv()
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    def load_pdf(self):
        pdf_text = ""
        with fitz.open(self.pdf_path) as pdf_document:
            for page in pdf_document:
                pdf_text += page.get_text()
        return pdf_text

    def split_text(self, pdf_text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(pdf_text)
        return splits

    def embed_texts(self, splits):
        self.vectorstore = Chroma.from_texts(texts=splits, embedding=OpenAIEmbeddings())
        self.retriever = self.vectorstore.as_retriever()

    def initialize_rag_chain(self):
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

        def format_texts(docs):
            text_contents = [doc.page_content for doc in docs]
            return "\n\n".join(text_contents)

        self.rag_chain = (
            {"context": self.retriever | format_texts, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def run_rag_system(self):
        while True:
            try:
                question = input("Question: ")
                response = self.rag_chain.invoke(question)
                print(response)
            except KeyboardInterrupt:
                print("\nCtrl+C detected. Exiting...")
                exit()

# Usage
if __name__ == "__main__":
    pdf_path = "./pdf/sys_design.pdf"
    rag_system = RAGSystem(pdf_path)
    pdf_text = rag_system.load_pdf()
    text_splits = rag_system.split_text(pdf_text)
    rag_system.embed_texts(text_splits)
    rag_system.initialize_rag_chain()
    rag_system.run_rag_system()
