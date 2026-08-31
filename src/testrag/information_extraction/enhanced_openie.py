import logging
import re
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List, TypedDict

from ..llm import CacheOpenAI
from ..prompts.prompt_template_manager import PromptTemplateManager
from .proposition_extraction import PropositionExtractor
from ..utils.misc_utils import PropositionRawOutput, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]

def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    eval_string = match.group()

    return eval(eval_string)["entities"]


class EnhancedOpenIE:
    """
    Enhanced version of OpenIE that uses proposition extraction before entity-relation extraction.
    This creates more contextually aware triples by first breaking passages into atomic propositions.
    """

    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={
                "system": "system",
                "user": "user",
                "assistant": "assistant",
            }
        )
        self.llm_model = llm_model
        self.proposition_extractor = PropositionExtractor(llm_model)

    def batch_openie(self, chunks: Dict[str, ChunkInfo], skip_triples=True) -> Tuple[Dict[str, NerRawOutput], Optional[Dict[str, TripleRawOutput]], Dict[str, PropositionRawOutput]]:
        """
        Conduct batch OpenIE with proposition extraction.
        
        Args:
            chunks: Dictionary of chunk IDs to chunk info
            skip_triples: If True, skip triple extraction and use only propositions
            
        Returns:
            Tuple of dictionaries with NER, (optionally) triple, and proposition extraction results
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        # Step 1: Extract named entities in batch
        ner_results_list = []

        # Process NER sequentially
        for chunk_key, passage in tqdm(chunk_passages.items(), desc="Extracting named entities"):
            ner_results_list.append(self.ner(chunk_key, passage))

        # Convert NER list to dictionary
        ner_results_dict = {res.chunk_id: res for res in ner_results_list}

        # Step 2: Extract propositions in batch using named entites
        # Create a dictionary mapping chunk IDs to named entities
        named_entities_dict = {chunk_key: ner_output.unique_entities for chunk_key, ner_output in ner_results_dict.items()}

        # Extract propositions with the named entities
        proposition_results_dict = self.proposition_extractor.batch_extract_propositions(chunks, named_entities_dict)

        if skip_triples:
            return ner_results_dict, None, proposition_results_dict


    def ner(self, chunk_key: str, passage: str, temperature=0.0, fix_attempt=False, use_cache=True) -> NerRawOutput:
        """
        Extract named entities from a passage.
        
        Args:
            chunk_key: Identifier for the chunk
            passage: The text passage to extract entities from
            
        Returns:
            NerRawOutput object containing the entities and metadata
        """

        ner_input_message = self.prompt_template_manager.render(name='ner_expanded'	, passage=passage)
        raw_response = ""

        try:
            # LLM INFERENCE
            while len(raw_response) == 0:
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=ner_input_message,
                    temperature=temperature,
                    use_cache=use_cache
                )
                if len(raw_response) == 0:
                    logger.warning("Empty response, try again.")
                    use_cache = False
            
            metadata['cache_hit'] = cache_hit

            if metadata['finish_reason'] == 'length':
                logger.warning("="*80)
                logger.warning(f"LENGTH LIMIT REACHED! Raw response: {raw_response}")
                logger.warning("="*80)

            real_response = raw_response

            if fix_attempt:
                logger.warning(f"Fix attempt with following response: {raw_response}")

            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

            if len(unique_entities) == 0:
                logger.warning(f"Number of entities is 0, raw_response: {raw_response}, passage: {passage}")

            if temperature > 0.0:
                logger.info(f"Fixed with following response: {raw_response} at temperature {temperature}")

        except Exception as e:
            logger.warning(f"Error occurred: {e}\nraw_response: {raw_response}\npassage: {passage}")
            logger.warning(f"Response length: {len(raw_response)}")
            logger.warning(f"Error occurred, initiating JSON fix!")


            json_fix_message = self.prompt_template_manager.render(name='fix_json', json=raw_response)
            raw_response, _, _ = self.llm_model.infer(
                messages=json_fix_message,
                temperature=temperature,
            )

            try:
                extracted_entities = _extract_ner_from_response(real_response)
                unique_entities = list(dict.fromkeys(extracted_entities))

                if len(unique_entities) == 0:
                    logger.warning(f"Number of entities is 0, raw_response: {raw_response}, passage: {passage}")

                logger.info(f"JSON fix successful! {raw_response}")
            except Exception as e:
                logger.warning(f"JSON fix failed!")

                metadata.update({"error": str(e)})
                return NerRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    unique_entities=[],
                    metadata=metadata
                )
        
        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )