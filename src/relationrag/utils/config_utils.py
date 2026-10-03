import os
import logging

from dataclasses import dataclass, field
from typing import Optional, Literal

logger = logging.getLogger(__name__)

@dataclass
class BaseConfig:
    # LLM specific attributes
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI-compatible LLM API. If None, the OpenAI client default is used."}
    )

    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_return_as_cpu: bool = field(
        default=True,
        metadata={"help": "Whether to move encoded embeddings to CPU before returning."}
    )
    embedding_return_as_numpy: bool = field(
        default=True,
        metadata={"help": "Whether to return encoded embeddings as NumPy arrays."}
    )

    # Graph construction specific attributes
    is_directed_graph: bool = field(
        default=True,
        metadata={"help": "Whether the graph is directed or not."}
    )

    # Seed proposition retrieval
    seed_k: int = field(
        default=20,
        metadata={"help": "Number of seed propositions to keep after nearest-neighbor search."}
    )
    lambda_mmr: float = field(
        default=1.0,
        metadata={"help": "MMR trade-off. 1.0 keeps the k most similar hits (no diversity); < 1 fetches 3k then MMR-selects k."}
    )

    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )

    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )

    def __post_init__(self):
        if self.save_dir is None:
            if self.dataset is None: self.save_dir = "outputs" # Running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # Customize your dataset's output dir here

        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            logger.error("OPENAI_API_KEY is not set")
            raise ValueError("OPENAI_API_KEY is not set")

        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")