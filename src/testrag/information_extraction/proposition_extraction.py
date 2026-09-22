import re
import ast
import json
import logging
from tqdm import tqdm
from typing import Dict, Optional, List

from ..prompts.prompt_template_manager import PromptTemplateManager
from ..utils.misc_utils import PropositionRawOutput
from ..utils.relation_utils import sanitize_proposition_relations

logger = logging.getLogger(__name__)


# Regex to find the structure, capturing text and entities content
# Using re.DOTALL flag later to handle multi-line text fields
REGEX_PATTERN = r'{\s*"text":\s*"(?P<text>.*?)"\s*,\s*"entities":\s*\[(?P<entities_content>.*?)]\s*}'

def fix_and_format_json_segment(original_segment: str, text_content: str, entities_content: str) -> str:
    """
    Attempts to fix unescaped quotes in text_content based on entities_content.
    Returns a valid, formatted JSON string for the segment if successful,
    otherwise returns the original_segment.
    """
    logger.info(f"Attempting fix for segment: {original_segment}")

    # --- 1. Safely evaluate entities structure ---
    try:
        # Construct a valid list string for ast.literal_eval
        entities_str_to_eval = f"[{entities_content.strip()}]" if entities_content.strip() else "[]"
        entities_list = ast.literal_eval(entities_str_to_eval)
        if not isinstance(entities_list, list) or not all(isinstance(e, str) for e in entities_list):
             raise ValueError("Evaluated entities is not a list of strings")
        # logging.info(f"  Successfully evaluated entities structure: {entities_list}")
    except (SyntaxError, ValueError, TypeError) as e:
        logger.warning(f"  Could not evaluate entities structure: {e}. Cannot perform fix.")
        return original_segment # Return original if entities structure is bad

    # --- 2. Store entity texts (no parsing needed now) ---
    entity_texts = set(entities_list) # Directly use the strings

    # --- 3. Find indices of *unescaped* double quotes in text_content ---
    unescaped_quote_indices = []
    i = 0
    while i < len(text_content):
        if text_content[i] == '"':
            # Check if it's escaped (look behind)
            if i > 0 and text_content[i-1] == '\\':
                # It's escaped, do nothing special
                pass
            else:
                # It's unescaped
                unescaped_quote_indices.append(i)
                # logging.debug(f"  Found unescaped quote at index {i}") # Use debug level
        i += 1 # Always advance index

    if not unescaped_quote_indices:
        # logging.info("  No unescaped quotes found in text. Re-formatting as JSON.")
        # Even if no quotes to fix, re-format to ensure valid JSON output
        try:
            output_dict = {"text": text_content, "entities": entities_list}
            return json.dumps(output_dict, ensure_ascii=False)
        except Exception as json_e:
             logger.error(f"  Failed to format segment as JSON even without quote fixing: {json_e}. Returning original.")
             return original_segment
    # logging.info(f"  Found {unescaped_quote_indices} to fix.")

    # --- 4. Apply fixing logic ---
    replacements_made = {} # Store index -> replacement char
    entity_lens = [len(entity) for entity in entity_texts]

    for uq_index in unescaped_quote_indices:
        for entity_len in entity_lens:
            # Check if the unescaped quote is EXACTLY at the start or end boundary
            if uq_index+1+entity_len <= len(text_content) and text_content[uq_index+1:uq_index+1+entity_len] in entity_texts:
                entity_end_index = uq_index+1+entity_len
                # Check the char at the OTHER end (the entity's end)
                if 0 <= entity_end_index < len(text_content):
                    other_end_char = text_content[entity_end_index]
                    if other_end_char == "'":
                        replacements_made[uq_index] = "'"
                    elif other_end_char == '"': # Escaped or unescaped double quote
                        break

            elif uq_index - entity_len - 1 >= 0 and text_content[uq_index - entity_len:uq_index] in entity_texts:
                entity_start_index = uq_index - entity_len - 1
                # Check the char at the OTHER end (the entity's start)
                if 0 <= entity_start_index < len(text_content):
                    other_end_char = text_content[entity_start_index]
                    if other_end_char == "'":
                        replacements_made[uq_index] = "'"
                    elif other_end_char == '"': # Escaped or unescaped double quote
                        break

    # Rebuild the string using the replacements map
    new_text_parts = []
    last_index = 0
    for index in sorted(replacements_made.keys()):
        new_text_parts.append(text_content[last_index:index])
        new_text_parts.append(replacements_made[index])
        last_index = index + 1 # Move past the original quote position
    new_text_parts.append(text_content[last_index:]) # Add the rest of the string

    fixed_text_content = "".join(new_text_parts)
    # logging.info(f"  Fixed text content: {fixed_text_content}")

    # --- 6. Reconstruct the segment as valid JSON ---
    try:
        output_dict = {
            "text": fixed_text_content,
            "entities": entities_list # Use the already validated list
        }
        # Produce compact JSON output, ensure_ascii=False handles unicode correctly
        reconstructed_segment = json.dumps(output_dict, ensure_ascii=False)
        # logging.info(f"  Successfully fixed and reconstructed JSON segment.")
        return reconstructed_segment
    except Exception as json_e:
        logger.error(f"  Failed to reconstruct segment as valid JSON after fixing: {json_e}. Returning original.")
        logger.error(f"  Original Text Content: {text_content}")
        logger.error(f"  Fixed Text Content Attempted: {fixed_text_content}")
        logger.error(f"  Entities List: {entities_list}")
        return original_segment # Fallback

def fix_large_json_text(large_text: str) -> str:
    """
    Finds all {"text":..., "entities":...} structures in large_text.
    Attempts to fix segments that are not valid JSON.
    Returns the full text with fixed segments replaced.
    """
    processed_parts = []
    last_end = 0
    regex = re.compile(REGEX_PATTERN, flags=re.DOTALL)

    for match in regex.finditer(large_text):
        match_start, match_end = match.span()
        original_segment = match.group(0)
        text_content = match.group('text')
        entities_content = match.group('entities_content')

        # Add the text *before* this match
        processed_parts.append(large_text[last_end:match_start])

        # --- Check if the original segment is valid JSON ---
        try:
            # Attempt to load the original segment directly
            json.loads(original_segment)
            processed_parts.append(original_segment) # Append original if valid
        except json.JSONDecodeError as e:
            logger.warning(f"Segment at {match_start} is NOT valid JSON: {e}. Attempting fix...")
            # --- If not valid, attempt to fix it ---
            fixed_segment = fix_and_format_json_segment(
                original_segment, text_content, entities_content
            )
            processed_parts.append(fixed_segment) # Append fixed or original (if fix failed)
        except Exception as e:
             logger.error(f"Unexpected error checking/fixing segment at {match_start}: {e}. Keeping original.")
             processed_parts.append(original_segment) # Append original on unexpected error


        last_end = match_end # Update the end position for the next iteration

    # Add any remaining text after the last match
    processed_parts.append(large_text[last_end:])

    # logging.info("Text processing finished.")
    return "".join(processed_parts)

class PropositionExtractor:
    """
    Class to extract propositions from passages before entity-relation extraction.
    Each proposition represents a fully contextualized unit of meaning.
    """

    def __init__(self, llm_model):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={
                "system": "system",
                "user": "user",
                "assistant": "assistant",
            }
        )
        self.llm_model = llm_model

    
    def batch_extract_propositions(self, chunks: Dict[str, Dict], named_entities_dict: Optional[Dict[str, List[str]]]=None) -> Dict[str, PropositionRawOutput]:
        """
        Extract propositions from multiple chunks in parallel.
        
        Args:
            chunks: Dictionary of chunk IDs to chunk info
            named_entities_dict: Optional dictionary mapping chunk IDs to pre-extracted named entities
            
        Returns:
            Dictionary of chunk IDs to proposition extraction results
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        proposition_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        pbar = tqdm(chunk_passages.items(), desc="Extracting propositions")

        for chunk_key, passage in pbar:
            named_entities = named_entities_dict.get(chunk_key, None) if named_entities_dict else None
            result = self.extract_propositions(chunk_key, passage, named_entities)
            proposition_results_list.append(result)
            metadata = result.metadata
            total_prompt_tokens += metadata.get('prompt_tokens', 0)
            total_completion_tokens += metadata.get('completion_tokens', 0)
            if metadata.get('cache_hit'):
                num_cache_hit += 1
            pbar.set_postfix({
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens,
                'num_cache_hit': num_cache_hit,
            })

        # Convert list of results to dictionary keyed by chunk ID
        proposition_results_dict = {res.chunk_id: res for res in proposition_results_list}

        return proposition_results_dict

    def extract_propositions(self, chunk_key: str, passage: str, named_entities: Optional[List[str]]=None, temperature=0.0) -> PropositionRawOutput:
        """
        Extract propositions from a passage.
        
        Args:
            chunk_key: Identifier for the chunk
            passage: The text passage to extract propositions from
            named_entities: Optional list of pre-extracted named entities to use
            
        Returns:
            PropositionRawOutput object containing the propositions and metadata
        """

        # Create the prompt for proposition extraction
        if named_entities:
            # Use the new prompt template with named entities
            proposition_input_message = self.prompt_template_manager.render(
                name='proposition_extraction',
                passage=passage,
                named_entities=json.dumps(named_entities)
            )
        else:
            proposition_input_message = self.prompt_template_manager.render(
                name='proposition_extraction',
                passage=passage,
                named_entities='[]'
            )
    
        raw_response = ""
        metadata = {}
        propositions = []
        relations = []

        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=proposition_input_message,
                temperature=temperature,
            )
            metadata['cache_hit'] = cache_hit

            if metadata['finish_reason'] == 'length':
                logger.warning("="*80)
                logger.warning(f"LENGTH LIMIT REACHED! Raw response: {raw_response}")
                logger.warning("="*80)
            real_response = raw_response

            # Extract proposition from the response
            extracted_data = self._extract_proposition_from_response(real_response)
            propositions = extracted_data["propositions"]
            relations = extracted_data["relations"]
        except ValueError as e:
            logger.warning(e)
            logger.warning(f"JSON parsing error! Try to fix JSON: {raw_response}")
            fix_json = True

            while fix_json:
                json_fix_message = self.prompt_template_manager.render(name='fix_json', json=raw_response)
                raw_response, _, _ = self.llm_model.infer(
                    messages=json_fix_message,
                    temperature=temperature,
                )
                try:
                    extracted_data = self._extract_proposition_from_response(raw_response)
                    propositions = extracted_data["propositions"]
                    relations = extracted_data["relations"]
                    fix_json = False
                    logger.info(f"JSON fix successful! {raw_response}")
                except Exception as e:
                    logger.warning(f"JSON fix error for chunk {chunk_key}: {e}")
                    logger.warning(f"Raw response: {raw_response}")
                    logger.warning(f"Try again!")
        
        except AttributeError as e:
            logger.warning(e)
            return self.extract_propositions(chunk_key, passage, named_entities, temperature=temperature)
        except AssertionError as e:
            logger.warning(f"Entities and text fields do not match, try to regenerate it: {raw_response}")
            return self.extract_propositions(chunk_key, passage, named_entities, temperature=temperature)
        except Exception as e:
            logger.warning(f"Unknown error for chunk {chunk_key}: {e}")
            logger.warning(f"Raw response: {raw_response}")
            metadata.update({"error": str(e)})
            return PropositionRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                propositions=[],
                relations=[],
                metadata=metadata
            )

        return PropositionRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            propositions=propositions,
            relations=relations,
            metadata=metadata
        )

    def _extract_proposition_from_response(self, response: str) -> Dict:
        """
        Extract propositions from the LLM response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Dictionary containing the extracted propositions and relations
        """

        # Extract the outermost JSON object, regardless of key order.
        start_idx = response.find("{")
        if start_idx < 0:
            raise ValueError(f"JSON response is invalid: {response}")
        curly_braces_count = 0

        in_quote = False
        escape = False
        idx = start_idx
        response_len = len(response)

        while idx < response_len:
            char = response[idx]
            if escape:
                if char == 'u': idx += 5
                else: idx += 1
                escape = False
                continue
            if char == '{' and not in_quote:
                curly_braces_count += 1
            elif char == '}' and not in_quote:
                curly_braces_count -= 1
            elif char == '"':
                in_quote = not in_quote
            elif char == '\\':
                escape = True
            if curly_braces_count == 0:
                break
            idx += 1
        if curly_braces_count > 0: # Incomplete Suspect
            if response.count('{') != response.count('}'): # Likely a mismatched quote issue
                raise AttributeError(f"JSON response is incomplete: {response}")
            else: # revert idx to the last curly brace
                idx = response.rfind('}', 0, idx)
        json_str = response[start_idx:idx+1]

        try:
            loaded_json = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = fix_large_json_text(json_str)
            loaded_json = json.loads(json_str)
            logger.info(f"JSON fix successful!")
        propositions = loaded_json["propositions"]
        for prop in propositions:
            assert "text" in prop and "entities" in prop
        relations = sanitize_proposition_relations(
            loaded_json.get("relations", []),
            num_propositions=len(propositions),
        )
        return {"propositions": propositions, "relations": relations}
