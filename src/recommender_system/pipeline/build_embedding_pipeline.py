import os
from recommender_system.utils.logger import get_logger
from recommender_system.utils.custom_exception import CustomException

from recommender_system.data_loader import AnimeDataLoader
from recommender_system.vector_store import VectorStoreBuilder

logger = get_logger(__name__)

RAW_DATA_PATH = "data/anime_with_synopsis.csv"
PROCESSED_DATA_PATH = "data/anime_processed.csv"
PERSIST_DIR = "chroma_db"


def main():
    try:
        logger.info("Starting the embedding build pipeline...")

        # 1. Load and process raw data
        logger.info("Loading and processing raw data...")
        loader = AnimeDataLoader(
            original_csv=RAW_DATA_PATH,
            processed_csv=PROCESSED_DATA_PATH
        )
        processed_csv = loader.load_and_process()
        logger.info(f"Processed CSV saved at: {processed_csv}")

        # 2. Build vector store
        logger.info("Building vector store...")
        vector_builder = VectorStoreBuilder(
            csv_path=processed_csv,
            persist_dir=PERSIST_DIR
        )
        vector_builder.build_and_save_vectorstore()
        logger.info(f"Vector store created at: {PERSIST_DIR}")

        logger.info("Embedding build pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise CustomException("Error in build embedding pipeline", e)


if __name__ == "__main__":
    main()
