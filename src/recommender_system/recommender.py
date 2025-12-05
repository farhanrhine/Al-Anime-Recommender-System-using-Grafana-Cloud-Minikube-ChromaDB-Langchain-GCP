from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from recommender_system.prompt_template import get_anime_prompt
from recommender_system.config.settings import GROQ_API_KEY, MODEL_NAME

from recommender_system.utils.logger import get_logger
from recommender_system.utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommender:
    def __init__(self, vectorstore):
        try:
            logger.info("Initializing Anime Recommender (docs-aligned)...")

            self.llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=0)

            # official: convert vector store -> retriever (Runnable)
            self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

            # prompt: use your PromptTemplate helper (make sure it's a langchain_core prompt)
            self.prompt = get_anime_prompt()

            # official LCEL / runnable composition
            self.rag_pipeline = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
            )

        except Exception as e:
            raise CustomException("Failed to initialize AnimeRecommender", e)

    def get_recommendation(self, query: str) -> str:
        try:
            result = self.rag_pipeline.invoke(query)
            # depending on the LLM wrapper the output may be a string or object
            # for Groq wrappers it's commonly a text or .content — adapt if needed
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            raise CustomException("Failed to generate recommendation", e)
