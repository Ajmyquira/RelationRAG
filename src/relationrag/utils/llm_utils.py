import re
from typing import TypedDict, Union, Optional
from string import Template

class TextChatMessage(TypedDict):
    """Representation of a single text-based chat message in the chat history."""
    role: str  # Either "system", "user", or "assistant"
    content: Union[str, Template]  # The text content of the message (could also be a string.Template instance)

def convert_format_to_template(original_string: str, placeholder_mapping: Optional[dict] = None, static_values: Optional[dict] = None) -> str:
    """
    Converts a .format() style string to a Template-style string.

    Args:
        original_string (str): The original string using .format() placeholders.
        placeholder_mapping (dict, optional): Mapping from original placeholder names to new placeholder names.
        static_values (dict, optional): Mapping from original placeholders to static values to be replaced in the new template.

    Returns:
        str: The converted string in Template-style format.
    """

    # Initialize mappings
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}

    # Regular expression to find .format() style placeholders
    placeholder_pattern = re.compile(r'\{(\w+)\}')
    
    # Subtitute placeholders in the string
    def replace_placeholders(match):
        original_placeholder = match.group(1)

        # If the placeholder is in static_values, substitute this value directly
        if original_placeholder in static_values:
            return str(static_values[original_placeholder])

        # Otherwise, rename the placeholder if needed, or keep it as is
        new_placeholder = placeholder_mapping.get(original_placeholder, original_placeholder)
        return f'${{{new_placeholder}}}'

    # Replace all placeholders
    template_string = placeholder_pattern.sub(replace_placeholders, original_string)

    return template_string