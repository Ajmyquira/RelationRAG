import os
import json
import logging

from dataclasses import asdict
from typing import Dict, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

import igraph as ig

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .utils.misc_utils import (
    NerRawOutput,
    PropositionRawOutput,
    compute_mdhash_id,
    extract_proposition_entities,
    flatten_propositions,
)
from .utils.config_utils import BaseConfig
from .utils.entity_synonymy import find_synonym_pairs
from .utils.relation_utils import sanitize_proposition_relations
from .information_extraction.enhanced_openie import EnhancedOpenIE


logger = logging.getLogger(__name__)

class TestRAG:
    def __init__(self, global_config=None, save_dir=None, llm_model_name=None, embedding_model_name=None, llm_base_url=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance.
                A instance of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific PropRAG instances will be stored.
                This defaults to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (EnhancedOpenIE): The Open Information Extraction module.
            graph: The graph instance initialized by the `initialize_graph` method.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed vLLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
            print("BaseConfig!")
        else:
            self.global_config = global_config
            print("GlobalConfig!")
        print("global_config: ", self.global_config)
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        _print_config = ",\n ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"TestRAG init with config:\n {_print_config}\n")

        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        logger.info("Using EnhancedOpenIE with proposition extraction")
        self.llm_model: BaseLLM = _get_llm_class(self.global_config)
        self.openie = EnhancedOpenIE(llm_model=self.llm_model)

        self.graph = self.initialize_graph()
        self.chunk_to_doc_id = {}
        self.typed_edges = set()

        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(self.global_config)
        self.chunk_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "chunk_embeddings"),
            self.global_config.embedding_batch_size,
            'chunk'
        )
        self.entity_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "entity_embeddings"),
            self.global_config.embedding_batch_size,
            'entity'
        )
        self.proposition_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "proposition_embeddings"),
            self.global_config.embedding_batch_size,
            'proposition'
        )

        self.openie_results_path = os.path.join(
            self.global_config.save_dir,
            f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json'
        )

    def initialize_graph(self):
        """
        Initializes a graph using a GraphML file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a GraphML file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """

        self._graphml_xml_file = os.path.join(self.working_dir, "graph.graphml")

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graphml_xml_file):
                preloaded_graph = ig.Graph.Read_GraphML(self._graphml_xml_file)
            
        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges.")
            return preloaded_graph

    def index(self, docs: list[dict]):
        """
        Indexes the given documents into an entity–proposition–passage knowledge graph
        and encodes passages, entities and propositions separately for later retrieval.

        Parameters:
            docs : list[dict]
                Documents to index. Each dict should include `title` and `text`.
        """
        
        logger.info("Indexing documents")
        logger.info("Perfoming OpenIE")

        passages, doc_ids = self._normalize_docs(docs)
        self.chunk_embedding_store.insert_strings(passages)
        chunks = self.chunk_embedding_store.get_text_for_all_rows()

        self.chunk_to_doc_id = {}
        for passage, doc_id in zip(passages, doc_ids):
            chunk_key = compute_mdhash_id(passage, prefix="chunk-")
            self.chunk_to_doc_id[chunk_key] = doc_id

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k: chunks[k] for k in chunk_keys_to_process}

        # Extract named entities and propositions
        if len(chunk_keys_to_process) > 0:
            logger.info("Running EnhancedOpenIE in index")
            new_ner_results_dict, new_proposition_results_dict = self.openie.batch_openie(new_openie_rows)

            self.merge_openie_results(
                all_openie_info,
                new_openie_rows,
                new_ner_results_dict,
                new_proposition_results_dict
            )
        
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        # Sanity check
        assert len(chunks) == len(all_openie_info), f"{len(chunks)} chunks but {len(all_openie_info)} OpenIE results"

        # Encoding entities and propositions
        chunk_ids = list(chunks.keys())
        logger.info(f"Processing proposition for embedding")
        proposition_results_dict = {}

        self.proposition_to_passages = defaultdict(set)

        for chunk_item in all_openie_info:
            chunk_id = chunk_item['idx']
            proposition_results_dict[chunk_id] = PropositionRawOutput(
                chunk_id=chunk_id,
                response="",
                metadata={},
                propositions=chunk_item['propositions'],
                relations=chunk_item['relations'],
            )
            for prop in chunk_item['propositions']:
                prop_text = prop["text"]
                prop_key = compute_mdhash_id(prop_text, prefix="proposition-")
                self.proposition_to_passages[prop_key].add(chunk_id)

        chunk_propositions_list = [proposition_results_dict[chunk_id].propositions for chunk_id in chunk_ids]
        chunk_relations_list = [proposition_results_dict[chunk_id].relations for chunk_id in chunk_ids]
        entity_nodes = extract_proposition_entities(chunk_propositions_list)
        propositions_flat = flatten_propositions(chunk_propositions_list)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Propositions")
        self.proposition_embedding_store.insert_strings([prop["text"] for prop in propositions_flat])

        # Records which entities belong to each proposition
        self.proposition_to_entities_map = {}
        for prop in propositions_flat:
            if "text" in prop and "entities" in prop:
                prop_text = prop["text"]
                prop_key = compute_mdhash_id(prop_text, prefix="proposition-")
                self.proposition_to_entities_map[prop_key] = prop["entities"]
            else:
                logger.warning(f"Skipping proposition without required fields: {prop}")

        props_with_passages = sum(len(self.proposition_to_passages[prop]) for prop in self.proposition_to_passages if self.proposition_to_passages[prop])
        total_props = len(self.proposition_to_entities_map)
        avg_passages = sum(len(passages) for passages in self.proposition_to_passages.values()) / max(1, len(self.proposition_to_passages))            

        logger.info(f"Built proposition-passage map: {props_with_passages}/{total_props} propositions mapped.")
        logger.info(f"Average passages per proposition: {avg_passages:.2f}")

        logger.info(f"Constructing Graph")

        self.typed_edges = set()

        logger.info(f"Using entity-proposition-passage graph construction")
        self.add_proposition_entity_edges() # Add edges between propositions and entities ('menciona')
        self.add_passage_proposition_edges(chunk_ids, chunk_propositions_list) # Add edges between passage and proposition ('contiene')
        self.add_proposition_proposition_edges(chunk_ids, chunk_propositions_list, chunk_relations_list) # Add directed edges between propositions within each chunk

        self.add_synonymy_edges() # Add bidirectional 'sinonimo' edges between entity aliases
        self.augment_graph() # Add new nodes and edges to the graph
        self.save_igraph() # Save the graph to a GraphML file


    def _normalize_docs(self, docs: list[dict]) -> Tuple[list[str], list[str]]:
        """
        Return (passage texts, doc ids) for indexing. One corpus item = one passage.
        Assumes docs is a list of dicts with 'title' and 'text' or 'content'.
        """
        passages = []
        doc_ids = []
        for doc in docs:
            title = doc["title"]
            passages.append(f"{title}\n{doc['text']}")
            doc_ids.append(title)
        return passages, doc_ids
   

    def load_existing_openie(self, chunk_keys: list[str]) -> Tuple[list[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (list[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[list[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """
        
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)

        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys
        
        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(
        self,
        all_openie_info: list[dict],
        chunks_to_save: Dict[str, dict],
        ner_results_dict: Dict[str, NerRawOutput],
        proposition_results_dict: Dict[str, PropositionRawOutput],
    ):
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the named-entity recognition (NER) entities, propositions and
        proposition-proposition relations with their respective text passages using the provided
        chunk keys. The resulting merged data is appended to the `all_openie_info` list containing
        dictionaries with combined and organized data for further processing or storage.

        Parameters:
            all_openie_info (list[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            proposition_results_dict (Dict[str, PropositionRawOutput]): A dictionary mapping chunk
                keys to their corresponding proposition extraction results.

        Returns:
            list[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {
                'idx': chunk_key,
                'passage': passage,
                'doc_id': self.chunk_to_doc_id[chunk_key],
                'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                'propositions': proposition_results_dict[chunk_key].propositions,
                'relations': proposition_results_dict[chunk_key].relations,
            }

            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: list[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : list[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        for chunk in all_openie_info:
            for i, e in enumerate(chunk['extracted_entities']):
                if not isinstance(e, str):
                    logger.warning(f"Non-string entity {e!r} in chunk {chunk['idx']}, casting to str")
                chunk['extracted_entities'][i] = str(e)
        sum_phrase_chars = sum(len(e) for chunk in all_openie_info for e in chunk['extracted_entities'])
        sum_phrase_words = sum(len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities'])
        num_phrases = sum(len(chunk['extracted_entities']) for chunk in all_openie_info)

        if num_phrases == 0:
            logger.warning("No entities extracted, OpenIE averages set to 0")

        openie_dict = {
            'docs': all_openie_info,
            'avg_ent_chars': round(sum_phrase_chars / num_phrases, 4) if num_phrases else 0,
            'avg_ent_words': round(sum_phrase_words / num_phrases, 4) if num_phrases else 0,
        }
        with open(self.openie_results_path, 'w') as f:
            json.dump(openie_dict, f)
        logger.info(f"OpenIE results saved to {self.openie_results_path}")


    def record_edge(self, source_id: str, target_id: str, rel_type: str, symmetric: bool = False):
        """Record a typed directed edge. Duplicate (src, type, dst) triples are ignored."""

        if source_id == target_id: return

        self.typed_edges.add((source_id, target_id, rel_type))
        if symmetric:
            self.typed_edges.add((target_id, source_id, rel_type))

    def add_proposition_entity_edges(self):
        """
        Connect each proposition to the entity nodes it mentions (`menciona`).

        Note: This method only collects relationships in typed_edges.
        Actual vertices and edges are added later by augment_graph().
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info("Connecting proposition nodes to entity nodes")

        for prop_key, entities in tqdm(self.proposition_to_entities_map.items(), desc="Adding proposition-entity edges"):
            if prop_key not in current_graph_nodes:
                for entity_text in entities:
                    entity_key = compute_mdhash_id(entity_text, prefix="entity-")
                    self.record_edge(prop_key, entity_key, "menciona")

        logger.info("Finished adding proposition-entity edges")

    def add_passage_proposition_edges(self, chunk_ids: list[str], chunk_propositions_list: list[list[Dict]]):
        """
        Connect each new passage (chunk) node to the proposition nodes extracted from it (`contiene`).

        Parameters:
            chunk_ids : list[str]
                Identifiers of passage nodes.
            chunk_propositions_list : list[list[Dict]]
                List of propositions extracted from each chunk.
        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info("Connecting passage nodes to proposition nodes")

        for idx, chunk_key in tqdm(enumerate(chunk_ids), desc="Adding passage-proposition edges"):
            if chunk_key not in current_graph_nodes:
                for prop in chunk_propositions_list[idx]:
                    prop_key = compute_mdhash_id(prop["text"], prefix="proposition-")
                    self.record_edge(chunk_key, prop_key, "contiene")

        logger.info("Finished adding passage-proposition edges")

    def add_proposition_proposition_edges(
        self,
        chunk_ids: list[str],
        chunk_propositions_list: list[list[Dict]],
        chunk_relations_list: list[list[Dict]],
    ):
        """Add intra-document proposition-proposition edges from joint extraction."""

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info("Connecting proposition nodes to proposition nodes")
        added = 0

        for chunk_key, propositions, relations in zip(chunk_ids, chunk_propositions_list, chunk_relations_list):
            if chunk_key in current_graph_nodes:
                continue
            prop_keys = [compute_mdhash_id(prop["text"], prefix="proposition-") for prop in propositions]
            cleaned = sanitize_proposition_relations(relations, num_propositions=len(prop_keys))
            for rel in cleaned:
                src_key = prop_keys[rel["src"]]
                dst_key = prop_keys[rel["dst"]]
                self.record_edge(src_key, dst_key, rel["type"], symmetric=rel["symmetric"])
                added += 1

        logger.info(f"Finished adding {added} proposition-proposition relations")

    def add_synonymy_edges(self):
        """Add bidirectional `sinonimo` edges between entity aliases using string identity."""

        logger.info("Expanding graph with entity synonymy edges")

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_texts = [row["content"] for row in self.entity_id_to_row.values()]
        pairs = find_synonym_pairs(entity_texts)

        added = 0
        for left, right in tqdm(pairs, desc="Adding synonymy edges"):
            left_key = compute_mdhash_id(left, prefix="entity-")
            right_key = compute_mdhash_id(right, prefix="entity-")
            if left_key in current_graph_nodes and right_key in current_graph_nodes:
                continue
            self.record_edge(left_key, right_key, "sinonimo", symmetric=True)
            added += 1

        logger.info(f"Finished adding {added} entity synonym pairs")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity, proposition and passage embedding stores
        based on their attributes.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_nodes = self.entity_embedding_store.get_text_for_all_rows()
        proposition_nodes = self.proposition_embedding_store.get_text_for_all_rows()
        passage_nodes = self.chunk_embedding_store.get_text_for_all_rows()

        for node_id, node in entity_nodes.items():
            node["name"] = node_id
            node["node_type"] = "entity"
            node["doc_id"] = ""
        for node_id, node in proposition_nodes.items():
            node["name"] = node_id
            node["node_type"] = "proposition"
            node["doc_id"] = ""
        for node_id, node in passage_nodes.items():
            node["name"] = node_id
            node["node_type"] = "passage"
            node["doc_id"] = self.chunk_to_doc_id.get(node_id, "")

        nodes = {}
        nodes.update(entity_nodes)
        nodes.update(proposition_nodes)
        nodes.update(passage_nodes)

        new_nodes = {}
        for node_id, node in nodes.items():
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """Materialize typed directed edges from `typed_edges` into the igraph object."""

        edge_source_nodes_keys = []
        edge_target_nodes_keys = []
        edge_types = []

        for src, tgt, rel_type in self.typed_edges:
            if src == tgt:
                continue
            edge_source_nodes_keys.append(src)
            edge_target_nodes_keys.append(tgt)
            edge_types.append(rel_type)

        valid_edges = []
        valid_attrs = {"type": []}
        current_node_ids = set(self.graph.vs["name"]) if self.graph.vcount() > 0 else set()
        for source_node_id, target_node_id, rel_type in zip(
            edge_source_nodes_keys, edge_target_nodes_keys, edge_types
        ):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                valid_attrs["type"].append(rel_type)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")

        if valid_edges:
            self.graph.add_edges(valid_edges, attributes=valid_attrs)

    def save_igraph(self):
        logger.info(f"Writting graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges")
        self.graph.write_graphml(self._graphml_xml_file)
        logger.info(f"Saving graph completed!")
