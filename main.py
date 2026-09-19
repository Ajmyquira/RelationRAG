import json
import argparse

from src.testrag import TestRAG
from src.testrag.utils.config_utils import BaseConfig
from src.testrag.utils.misc_utils import string_to_bool

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="Test RAG")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa", help="Dataset name")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openia.com/v1", help='LLM base URL')
    parser.add_argument("--embedding_name", type=str, default="facebook/contriever", help="Embedding model name")
    parser.add_argument("--force_index_from_scratch", type=str, default="false", help="If set to True, will ignore all existing storage files and graph data and will rebuild from scrath.")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false", help="If set to False, will try to first reuse openie result for the corpus if they exist.")

    args = parser.parse_args()

    dataset_name = args.dataset
    
    corpus_path = f"data/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [{"title": doc["title"], "text": doc["text"]} for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    config = BaseConfig(
        llm_name=args.llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
    )

    logging.basicConfig(level=logging.INFO)

    testrag = TestRAG(global_config=config)
    testrag.index(docs)

if __name__ == "__main__":
    main()
