import logging
import numpy as np

from argparse import ArgumentTypeError
from dataclasses import dataclass, field
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class PropositionRawOutput:
    chunk_id: str
    response: str
    propositions: list[Dict[str, Any]]  # List of proposition objects with text and entities
    metadata: Dict[str, Any] = None
    relations: list[Dict[str, Any]] = field(default_factory=list)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: list[str]
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