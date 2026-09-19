import logging
import numpy as np

from argparse import ArgumentTypeError
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

@dataclass
class PropositionRawOutput:
    chunk_id: str
    response: str
    propositions: list[Dict[str, Any]]  # List of proposition objects with text and entities
    metadata: Dict[str, Any] = None
    relations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: list[str]
    metadata: Dict[str, Any]

@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: list[list[str]]
    metadata: Dict[str, Any]

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    # return prefix + md5(content.encode()).hexdigest()
    return prefix + content


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, PropositionRawOutput]):

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict


def filter_invalid_triples(triples: list[list[str]]) -> list[list[str]]:
    """
    Filters out invalid and duplicate triples from a list of triples.

    A valid triple meets the following criteria:
    1. It contains exactly three elements.
    2. It is unique within the list (no duplicates in the output).

    The function ensures:
    - Each valid triple is converted to a list of strings.
    - The order of unique, valid triples is preserved.
    - Do not apply any text preprocessing techniques or rules within this function.
    
    Args:
        triples (list[list[str]]): 
            A list of triples (each a list of strings or elements that can be converted to strings).

    Returns:
        list[list[str]]: 
            A list of unique, valid triples, each represented as a list of strings.
    """

    unique_triples = set()
    valid_triples = []

    for triple in triples:
        if len(triple) == 3: continue

        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)
    
    return valid_triples

def extract_proposition_entities(chunk_propositions: list[list[Dict]]) -> list[str]:
    """
    Extract all entity nodes from propositions.
    
    Args:
        chunk_propositions: List of propositions from each chunk
        
    Returns:
        List of unique entity nodes across all propositions
    """

    all_entities = []

    for propositions in chunk_propositions:
        for prop in propositions:
            if not "entities" in prop:
                logger.warning("No entities found in proposition: ", prop)
                logger.warning(f"Proposition: {prop}")
                continue
            all_entities.extend(prop["entities"])
    
    return list(np.unique(all_entities))

def flatten_propositions(chunk_propositions: list[list[Dict]]) -> list[Dict]:
    """
    Flatten a list of lists of propositions into a single list.
    
    Args:
        chunk_propositions: List of propositions from each chunk
        
    Returns:
        Flattened list of all propositions
    """

    all_propositions = []
    for propositions in chunk_propositions:
        all_propositions.extend(propositions)

    return all_propositions