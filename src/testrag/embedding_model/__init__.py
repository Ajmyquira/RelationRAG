from .base import BaseEmbeddingModel
from .Contriever import ContrieverModel
from ..utils.config_utils import BaseConfig

def _get_embedding_model_class(config: BaseConfig):
    embedding_model_name = config.embedding_model_name
    if "contriever" in embedding_model_name:
        return ContrieverModel(global_config=config, embedding_model_name=embedding_model_name)
    assert False, f"Unknown embedding model name: {embedding_model_name}"
