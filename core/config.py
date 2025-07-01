import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings():
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    ASSISTANT_ID: Optional[str] = None

    # Allow extra fields, e.g., from environment variables that are not part of the model
    # model_config = SettingsConfigDict(extra='ignore')

settings = Settings()