import pandas as pd
from recommender_system.utils.logger import get_logger
from recommender_system.utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeDataLoader:
    """
    Loads and processes the anime CSV dataset:
    - Reads raw CSV
    - Validates required columns
    - Creates a combined_info column
    - Saves processed CSV
    """

    def __init__(self, original_csv: str, processed_csv: str):
        self.original_csv = original_csv
        self.processed_csv = processed_csv

    def load_and_process(self) -> str:
        try:
            logger.info(f"Loading dataset from: {self.original_csv}")
            df = pd.read_csv(
                self.original_csv,
                encoding='utf-8',
                on_bad_lines='skip'
            ).dropna()

            logger.info("CSV loaded successfully. Validating columns...")

            required_cols = {'Name', 'Genres', 'Synopsis'}
            missing = required_cols - set(df.columns)

            if missing:
                raise CustomException(
                    f"Missing required column(s): {missing}"
                )

            logger.info("Columns validated. Creating combined_info column...")

            df['combined_info'] = (
                "Title: " + df['Name'] +
                "\nGenres: " + df['Genres'] +
                "\nOverview: " + df['Synopsis']
            )

            df[['combined_info']].to_csv(
                self.processed_csv,
                index=False,
                encoding='utf-8'
            )

            logger.info(f"Processed CSV saved to: {self.processed_csv}")
            return self.processed_csv

        except Exception as e:
            raise CustomException("Failed during data loading & processing", e)


