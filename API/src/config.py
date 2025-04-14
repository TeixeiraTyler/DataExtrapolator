from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Data Extrapolator API"
    
    # ML settings
    Z_THRESHOLD: float = 3.0
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 0
    
    # Model parameters
    N_ESTIMATORS: list = [100, 200, 300]
    MAX_DEPTH: list = [10, 20, 30, None]
    MIN_SAMPLES_SPLIT: list = [2, 5, 10]
    
    class Config:
        env_file = ".env"
        case_sensitive = True 