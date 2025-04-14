from dependency_injector import containers, providers

from src.services.ml_service import MLService
from src.config import Settings


class Container(containers.DeclarativeContainer):
    """Dependency injection container."""
    
    config = providers.Singleton(Settings)
    
    ml_service = providers.Singleton(
        MLService,
        config=config
    ) 