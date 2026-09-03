import os
import json
import logging
import re

from dataclasses import asdict
from typing import Tuple, Set
from collections import defaultdict
from tqdm import tqdm

import igraph as ig

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .utils.misc_utils import *
from .utils.config_utils import BaseConfig
from .utils.embed_utils import retrieve_knn
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

        self._graphml_xml_file = os.path.join(
            self.working_dir,
            f"graph_{self.global_config.synonymy_edge_sim_threshold}.graphml"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graphml_xml_file):
                preloaded_graph = ig.Graph.Read_GraphML(self._graphml_xml_file)
            
        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges.")
            return preloaded_graph

    def index(self, docs: list[str]):
        """
        Indexes the given documents into an entity–proposition–passage knowledge graph
        and encodes passages, entities and propositions separately for later retrieval.

        Parameters:
            docs : list[str]
                A list of documents to be indexed.
        """
        
        logger.info("Indexing documents")
        logger.info("Perfoming OpenIE")

        self.chunk_embedding_store.insert_strings(docs)
        chunks = self.chunk_embedding_store.get_text_for_all_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k: chunks[k] for k in chunk_keys_to_process}

        # Extract named entities and propositions
        if len(chunk_keys_to_process) > 0:
            logger.info("Running EnhancedOpenIE in index")
            new_ner_results_dict, _, new_proposition_results_dict_props = self.openie.batch_openie(new_openie_rows)

            self.merge_openie_results(
                all_openie_info,
                new_openie_rows,
                new_ner_results_dict,
                None,
                new_proposition_results_dict_props
            )
        
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        # Sanity check
        ner_results_dict, proposition_results_dict_reformatted = reformat_openie_results(all_openie_info)
        print(len(chunks), len(ner_results_dict), len(proposition_results_dict_reformatted), len(all_openie_info))
        assert len(chunks) == len(ner_results_dict) == len(proposition_results_dict_reformatted)

        # Encoding entities and propositions
        chunk_ids = list(chunks.keys())
        logger.info(f"Processing proposition for embedding")
        proposition_results_dict = {}

        self.proposition_to_passages = defaultdict(set)

        for chunk_item in all_openie_info:
            chunk_id = chunk_item['idx']
            if 'propositions' in chunk_item:
                proposition_results_dict[chunk_id] = PropositionRawOutput(
                    chunk_id=chunk_id,
                    response="",
                    metadata={},
                    propositions=chunk_item['propositions']
                )
                for prop in chunk_item['propositions']:
                    if "text" in prop:
                        prop_text = prop["text"]
                        prop_key = compute_mdhash_id(prop_text, prefix="proposition-")
                        self.proposition_to_passages[prop_key].add(chunk_id)

        chunk_propositions_list = [proposition_results_dict[chunk_id].propositions for chunk_id in chunk_ids]
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

        self.node_to_node_stats = {} # (source_id, target_id) -> weight

        logger.info(f"Using entity-proposition-passage graph construction")
        self.add_entity_proposition_edges()
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_propositions_list)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges() # Similar entity names (KNN on embeddings)
            self.augment_graph() # add_new_nodes + add_new_edges -> igraph
            self.save_igraph() # GraphML


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
        proposition_results_dict_triples: Dict[str, PropositionRawOutput],
        proposition_results_dict_props: Dict[str, PropositionRawOutput] = None,
    ):
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including propositions (if available),
        named-entity recognition (NER) entities, and (legacy) propositions/triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (list[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            proposition_results_dict_triples (Dict[str, PropositionRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE (legacy) proposition/triple extraction results.
            proposition_results_dict_props (Dict[str, PropositionRawOutput], optional): A dictionary 
                mapping chunk keys to their corresponding main proposition extraction results.

        Returns:
            list[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {
                'idx': chunk_key,
                'passage': passage,
                'extracted_entities': ner_results_dict[chunk_key].unique_entities
            }

            if proposition_results_dict_triples is not None and chunk_key in proposition_results_dict_triples:
                chunk_openie_info['extracted_triples'] = proposition_results_dict_triples[chunk_key].propositions
            else:
                chunk_openie_info['extracted_triples'] = []

            if proposition_results_dict_props and chunk_key in proposition_results_dict_props:
                chunk_openie_info['propositions'] = proposition_results_dict_props[chunk_key].propositions

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
                    print(chunk)
                    print(e)
                chunk['extracted_entities'][i] = str(e)
        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0 and num_phrases > 0:
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': round(sum_phrase_chars / num_phrases, 4),
                'avg_ent_words': round(sum_phrase_words / num_phrases, 4)
            }
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")
        elif len(all_openie_info) > 0:
            logger.warning(f"No phrases extracted, cannot compute average for OpenIE results")
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': 0,
                'avg_ent_words': 0
            }
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")
        else:
            logger.warning(f"No OpenIE results to save")
            openie_dict = {
                'docs': [],
                'avg_ent_chars': 0,
                'avg_ent_words': 0
            }
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results (without averages) saved to {self.openie_results_path}")


    def record_edge(self, source_id: str, target_id: str, weight: float = 1.0):
        """ Record an edge in node_to_node_stats. Bidirectional if the graph is directed. """

        if source_id == target_id: return

        self.node_to_node_stats[(source_id, target_id)] = \
            self.node_to_node_stats.get((source_id, target_id), 0.0) + weight

        if self.global_config.is_directed_graph:
            self.node_to_node_stats[(target_id, source_id)] = \
                self.node_to_node_stats.get((target_id, source_id), 0.0) + weight

    def add_entity_proposition_edges(self):
        """
        Connect each entity to the proposition nodes that mention it.

        Note: This method only collects relationships in node_to_node_stats.
        Actual vertices and edges are added later by augment_graph().
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info("Connecting entity nodes to proposition nodes")

        for prop_key, entities in tqdm(self.proposition_to_entities_map.items(), desc="Adding entity-proposition edges"):
            if prop_key not in current_graph_nodes:
                for entity_text in entities:
                    entity_key = compute_mdhash_id(entity_text, prefix="entity-")
                    self.record_edge(entity_key, prop_key)

        logger.info("Finished adding entity-proposition edges")

    def add_passage_edges(self, chunk_ids: list[str], chunk_propositions_list: list[list[Dict]]):
        """
        Connect each new passage (chunk) node to the proposition nodes extracted from it.

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

        num_new_chunks = 0

        logger.info("Connecting proposition nodes to passage nodes")

        for idx, chunk_key in tqdm(enumerate(chunk_ids), desc="Adding proposition-passage edges"):
            if chunk_key not in current_graph_nodes:
                for prop in chunk_propositions_list[idx]:
                    if "text" not in prop: continue
                    prop_key = compute_mdhash_id(prop["text"], prefix="proposition-")
                    self.record_edge(prop_key, chunk_key)

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity 
        by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges.
        It first retrieves embeddings for all nodes, then conducts a nearest neighbor (KNN)
        search to find similar nodes. These similar nodes are identified based on a score threshold,
        and edges are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: 
                dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                contain `content` of entities used for comparison.
            entity_embedding_store:
                Manages retrieval of texts and embeddings for all rows related to entities.
            global_config:
                Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats:
                dict. Stores scores for edges between nodes representing their relationship.
        """

        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        query_node_key2knn_node_keys = retrieve_knn(
            query_ids=entity_node_keys,
            key_ids=entity_node_keys,
            query_vecs=entity_embs,
            key_vecs=entity_embs,
            k=self.global_config.synonymy_edge_topk,
            query_batch_size=self.global_config.synonymy_edge_query_batch_size,
            key_batch_size=self.global_config.synonymy_edge_key_batch_size,
            threshold_score=self.global_config.synonymy_edge_sim_threshold
        )

        num_synonym_proposition = 0
        # synonym_candidates = []

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys), desc="Adding synonymy edges"):
            # synonyms = []
            
            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0

                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        # synonyms.append((nn, score))
                        num_synonym_proposition += 1

                        self.node_to_node_stats[sim_edge] = self.node_to_node_stats.get(sim_edge, 0) + 1
                        num_nns += 1
            
            # synonym_candidates.append((node_key, synonyms))

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        # print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity, proposition and passage embedding stores
        based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity, proposition and passage embedding stores.
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_nodes = self.entity_embedding_store.get_text_for_all_rows()
        proposition_nodes = self.proposition_embedding_store.get_text_for_all_rows()
        passage_nodes = self.chunk_embedding_store.get_text_for_all_rows()

        nodes = entity_nodes
        nodes.update(proposition_nodes)
        nodes.update(passage_nodes)

        new_nodes = {}
        for node_id, node in nodes.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)


    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_nodes_keys = []
        edge_target_nodes_keys = []
        edge_metadata = []

        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight
            edge_source_nodes_keys.append(edge[0])
            edge_target_nodes_keys.append(edge[1])
            edge_metadata.append({"weight": weight})

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_nodes_keys, edge_target_nodes_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
            
        self.graph.add_edges(valid_edges, attributes=valid_weights)


    # def get_graph_info(self) -> Dict:
    #     """
    #     Obtains detailed information about the graph such as the number of nodes,
    #     propositions, and their classifications.

    #     This method calculates various statistics about the graph based on the
    #     stores and node-to-node relationships, including counts of phrase and
    #     passage nodes, total nodes, extracted propositions, propositions involving passage
    #     nodes, synonymy propositions, and total propositions.

    #     Returns:
    #         Dict
    #             A dictionary containing the following keys and their respective values:
    #             - num_phrase_nodes: The number of unique phrase nodes.
    #             - num_passage_nodes: The number of unique passage nodes.
    #             - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
    #             - num_extracted_propositions: The number of unique extracted propositions.
    #             - num_propositions_with_passage_node: The number of propositions involving at least one
    #               passage node.
    #             - num_synonymy_propositions: The number of synonymy propositions (distinct from extracted
    #               propositions and those with passage nodes).
    #             - num_total_propositions: The total number of propositions (edges).
    #     """
    #     graph_info = {}

    #     phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
    #     graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

    #     passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
    #     graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

    #     graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

    #     graph_info["num_extracted_propositions"] = len(self.proposition_embedding_store.get_all_ids())

    #     num_propositions_with_passage_node = 0
    #     passage_nodes_set = set(passage_nodes_keys)
    #     num_propositions_with_passage_node = sum(
    #         1 for node_pair in self.node_to_node_stats
    #         if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
    #     )
    #     graph_info['num_propositions_with_passage_node'] = num_propositions_with_passage_node

    #     graph_info['num_synonymy_propositions'] = len(self.node_to_node_stats) - graph_info[
    #         "num_extracted_propositions"] - num_propositions_with_passage_node

    #     graph_info["num_total_propositions"] = len(self.node_to_node_stats)

    #     return graph_info

    def save_igraph(self):
        logger.info(f"Writting graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges")
        self.graph.write_graphml(self._graphml_xml_file)
        logger.info(f"Saving graph completed!")
