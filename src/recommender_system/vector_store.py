import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from recommender_system.utils.logger import get_logger
from recommender_system.utils.custom_exception import CustomException
from recommender_system.config.settings import EMBEDDING_MODEL

logger = get_logger(__name__)
load_dotenv()

class VectorStoreBuilder:
    """
    Builds and loads the Chroma vector store using processed CSV.
    """

    def __init__(self, csv_path: str = None, persist_dir: str = "chroma_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def build_and_save_vectorstore(self) -> str:
        """
        Creates embeddings from processed CSV and saves them to ChromaDB.
        """
        try:
            if not os.path.exists(self.csv_path):
                raise CustomException(f"CSV file not found: {self.csv_path}")

            logger.info(f"Loading CSV for vector store: {self.csv_path}")

            loader = CSVLoader(
                file_path=self.csv_path,
                encoding="utf-8",
                metadata_columns=[]
            )
            data = loader.load()

            logger.info(f"Loaded {len(data)} documents. Splitting text...")

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = splitter.split_documents(data)

            logger.info(f"Created {len(texts)} text chunks. Building ChromaDB...")

            db = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding,
                persist_directory=self.persist_dir
            )

            db.persist()
            logger.info(f"Vector store saved at: {self.persist_dir}")

            return self.persist_dir

        except Exception as e:
            raise CustomException("Failed to build vector store", e)

    def load_vector_store(self):
        """
        Loads an existing Chroma vector store.
        """
        try:
            logger.info(f"Loading vector store from: {self.persist_dir}")

            if not os.path.exists(self.persist_dir):
                raise CustomException(
                    f"Persist directory does not exist: {self.persist_dir}"
                )

            db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding
            )

            logger.info("Vector store loaded successfully")
            return db

        except Exception as e:
            raise CustomException("Failed to load vector store", e)
