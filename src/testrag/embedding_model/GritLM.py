import torch
import numpy as np

from copy import deepcopy
from typing import (Optional, List)
from gritlm import GritLM

from ..utils.config_utils import BaseConfig

from .base import BaseEmbeddingModel, EmbeddingConfig

class GritLMEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)
        
        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
        
        self._init_embedding_model()

        self.embedding_model = GritLM(**self.embedding_config.model_init_params)
        self.embedding_dim = self.embedding_model.model.config.hidden_size

        self.device = self.embedding_model.device

    def _init_embedding_model(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        
        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "return_cpu": self.global_config.embedding_return_as_cpu,
            "return_numpy": self.global_config.embedding_return_as_numpy,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "model_name_or_path": self.embedding_model_name,
                "torch_dtype": "auto",
                "device_map": "auto", # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
                # "instruction": "",
            },
            "generate_params": {
                # "max_new_tokens": 
                # "do_sample": 
            }
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)

    def _get_formatted_instruction(self, instruction: str) -> str:
        return "<|user|>\n" + instruction + "\n<|embed|>" if instruction else "<|embed|>\n"
    
    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)
        if "instruction" in kwargs:
            params["instruction"] = self._get_formatted_instruction(params["instruction"])
        params["sentences"] = texts

        results = self.embedding_model.encode(**params)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results.T, axis=1)).T

        return results