import json
import argparse

from dotenv import load_dotenv
load_dotenv()

from src.relationrag import RelationRAG
from src.relationrag.utils.config_utils import BaseConfig
from src.relationrag.utils.misc_utils import string_to_bool

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

def index(args, config):
    corpus_path = f"data/{args.dataset}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [{"title": doc["title"], "text": doc["text"]} for doc in corpus]

    relationrag = RelationRAG(global_config=config)
    relationrag.index(docs)

def query(config):
    with open("question.txt", "r") as f:
        question = f.read().strip()
    if not question:
        raise SystemExit("error: question.txt is empty")

    relationrag = RelationRAG(global_config=config)
    print(relationrag.answer(question))

def main():
    parser = argparse.ArgumentParser(description="Relation RAG")
    parser.add_argument("--mode", type=str, choices=["index", "query"], default="index", help="index documents, or answer a question from an existing index")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa", help="Dataset name")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help='LLM base URL')
    parser.add_argument("--embedding_name", type=str, default="facebook/contriever", help="Embedding model name")
    parser.add_argument("--force_index_from_scratch", type=str, default="false", help="If set to True, will ignore all existing storage files and graph data and will rebuild from scrath.")
    parser.add_argument("--force_openie_from_scratch", type=str, default="false", help="If set to False, will try to first reuse openie result for the corpus if they exist.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = BaseConfig(
        llm_name=args.llm_name,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        llm_base_url=args.llm_base_url,
    )

    if args.mode == "index":
        index(args, config)
    else:
        query(config)

if __name__ == "__main__":
    main()
