import httpx
import logging
from copy import deepcopy
from packaging import version

from typing import List, Tuple

import openai
from openai import OpenAI
from .base import BaseLLM, LLMConfig
from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage

logger = logging.getLogger(__name__)

class CacheOpenAI(BaseLLM):
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        return cls(**global_config.__dict__)
    
    def __init__(
        self,
        llm_name: str = "gpt-4o-mini",
        api_key: str = None,
        llm_base_url: str = None,
        **kwargs
    ):
        super().__init__()
        self.llm_name = llm_name
        self.llm_base_url = llm_base_url

        self._init_llm_config(**kwargs)

        limits = httpx.Limits(max_connections=500, max_keepalive_connections=500)
        client = httpx.Client(limits=limits, timeout=httpx.Timeout(3*60, read=2*60))

        self.openai_client = OpenAI(base_url=self.llm_base_url, api_key=api_key, http_client=client)

    def _init_llm_config(self, **kwargs) -> None:
        config_dict = {
            "llm_name": self.llm_name,
            "llm_base_url": self.llm_base_url,
            "generate_params": {
                "model": self.llm_name,
                "max_completion_tokens": kwargs.get("max_new_tokens", 8192),
                "n": kwargs.get("num_gen_choices", 1),
                "seed": kwargs.get("seed", 0),
                "temperature": kwargs.get("temperature", 0.0),
            }
        }

        logger.info(f"LLM generate params: {config_dict['generate_params']}")
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[str, dict, bool]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        # logger.info(f"Calling OpenAI GPT API with: \n{params}")
        # params['extra_body'] = {"provider": {"allow_fallbacks": False, "order": ["Nebius"]}}

        # If we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"):
            params['max_tokens'] = params.pop('max_completion_tokens')
        response_checker = params.pop('response_checker', None)

        while True:
            try:
                response = self.openai_client.chat.completions.create(**params)
            except openai.APITimeoutError:
                logger.error(f"API timeout, try again: {params['messages']}")
                continue
            except Exception as e:
                logger.error(f"HTTP timeout, try again: {str(e)}")
                continue

            try:
                response_message = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
            except Exception as e:
                logger.error(f"Error getting response message, try again: {e}")
                logger.error(f"Response: {response}")
                logger.error(f"Error causing passage: {params['messages'][-1]}")
                continue

            if response_checker is not None and not response_checker(response_message, finish_reason, params):
                continue
            break
    
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason
        }

        return response_message, metadata, False
