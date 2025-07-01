import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    ASSISTANT_ID: str = None # Will be created and stored if not exists

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()