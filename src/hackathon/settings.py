from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_token: str = ""
    model_config = SettingsConfigDict(env_file=".env")


try:
    settings = Settings()
except ValidationError as e:
    print(
        "Some required environment variables are missing or invalid. "
        "Perhaps you have not set up your .env file correctly?"
    )
    raise e
