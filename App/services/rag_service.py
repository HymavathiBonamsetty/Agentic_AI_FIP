# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            show_progress=True
        )

        self.vector_store = Chroma(
            persist_directory="./rag_service_db",
            collection_name="teaching_assistant_collection",
            embedding_function=embeddings
        )

    def process_pdf_document(self, file_path: str=  "Assests/Let us c - Summary.pdf"):
        loader=PyPDFLoader(file_path)
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 150,
        )

        split_documents = splitter.split_documents(document)

        self.vector_store.add_documents(split_documents)


if __name__ == "__main__":
    service = EmbeddingService()
    service.process_pdf_document()