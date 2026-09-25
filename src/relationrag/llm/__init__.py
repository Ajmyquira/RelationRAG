from ..utils.config_utils import BaseConfig
from .openai_gpt import CacheOpenAI
from .base import BaseLLM

def _get_llm_class(config: BaseConfig):
    return CacheOpenAI.from_experiment_config(config)