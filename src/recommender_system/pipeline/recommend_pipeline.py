import os

from recommender_system.vector_store import VectorStoreBuilder
from recommender_system.recommender import AnimeRecommender

from recommender_system.utils.logger import get_logger
from recommender_system.utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    """
    Loads the existing Chroma vector store and provides a clean method
    for generating anime recommendations using the RAG-based recommender.
    """

    def __init__(self, persist_dir: str = "chroma_db"):
        try:
            logger.info("Initializing Recommendation Pipeline...")

            # 0. Validate vectorstore directory
            if not os.path.exists(persist_dir):
                raise CustomException(
                    f"Vector store path '{persist_dir}' not found. "
                    "Run build_embedding_pipeline.py first."
                )

            # 1. Load vector store
            vector_builder = VectorStoreBuilder(
                csv_path=None,
                persist_dir=persist_dir
            )
            vectorstore = vector_builder.load_vector_store()

            # 2. Pass vectorstore to AnimeRecommender
            self.recommender = AnimeRecommender(vectorstore)

            logger.info("Recommendation Pipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            raise CustomException("Error during pipeline initialization", e)

    def recommend(self, query: str, return_sources: bool = False):
        """
        Takes a user query and returns recommendations.
        Optionally returns source documents for debugging.
        """
        try:
            logger.info(f"Received user query: {query}")

            if not query or not isinstance(query, str):
                raise CustomException("Query must be a non-empty string.")

            response = self.recommender.get_recommendation(query)
            logger.info("Recommendation generated successfully.")

            # Note: get_recommendation returns just the answer string
            # Source documents are not exposed by the current recommender implementation
            if return_sources:
                logger.warning("return_sources=True requested, but sources not available with current implementation")
                return response, []

            return response

        except Exception as e:
            logger.error(f"Failed to get recommendation: {str(e)}")
            raise CustomException("Error while generating recommendation", e)
