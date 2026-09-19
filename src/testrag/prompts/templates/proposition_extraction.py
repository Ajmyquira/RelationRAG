from ...utils.llm_utils import convert_format_to_template

proposition_system = """Your task is to analyze text passages and break them down into precise, atomic propositions using a specified list of named entities, then list only the discourse or logical relations that hold between those propositions.

A proposition is a fully contextualized statement that expresses a single unit of meaning with complete specificity about the relationships described.

For each proposition:
1. Extract a complete, standalone statement that preserves the full context
2. Use ONLY the entities provided in the named_entities list - do not introduce new entities
3. Ensure each proposition contains only ONE claim or relationship
4. Be extremely specific about which entities are involved in each relationship
5. Maintain clear causal connections between related statements

Respond with a JSON object containing:
- "propositions": a list of objects with:
  - "text": The full proposition text as a complete, contextualized statement
  - "entities": An array of entities from the named_entities list that appear in that proposition
- "relations": a list of directed relations BETWEEN those propositions. Use 0-based indices into the propositions array. Include ONLY relations the passage clearly supports. If there is no discourse or logical nexus, omit the pair (do not invent a relation and do not use a type such as "none" or "similar"). Each relation object has:
  - "src": index of the source proposition
  - "dst": index of the target proposition
  - "type": a short snake_case identifier of 1-3 words, derived from the actual nexus (examples, not an exhaustive list: causes, supports, contradicts, modifies, precedes, exemplifies)
  - "symmetric": true only when the relation is symmetric (the same type holds in both directions, e.g. contradicts). false for directed relations such as causes or modifies. Do not treat an inverse type (causes vs is_caused_by) as symmetry.

Relation guidelines:
- Sharing entities does not forbid a relation and does not by itself justify one. The nexus cannot be mere similarity or co-mention.
- If several distinct nexuses hold between the same pair, emit all of them (different types).
- Direction: src is the proposition that exerts the nexus (the cause, the evidence, the qualifier).
- Do not duplicate the same (src, type, dst). If symmetric, emit one object with symmetric=true rather than both directions.

Critical Guidelines:
- ONLY use entities from the provided named_entities list
- Make relationships explicit and specific - clarify exactly which entities relate to which other entities
- Clarify precisely which entity a modifier applies to (e.g., specify which product had "80% improvement")
- Establish clear connections between related facts (e.g., "Adobe optimized their applications FOR THE M1 CHIP")
- Connect comparative statements to their specific reference points (e.g., "Adobe's applications on the M1 chip improved by 80% compared to Intel-based Macs")
- Preserve temporal information and causal relationships between events
- Make each proposition stand alone with all necessary context
- Include ALL relevant entities from the named_entities list in both the proposition text and entities array
- Ensure the collection of propositions captures ALL meaningful information in the passage

Example 1:
Passage: In 2020, after Apple launched the M1 chip, major software companies like Adobe optimized their applications, improving performance by up to 80% compared to Intel-based Macs.

Named entities: ["Apple", "M1 chip", "2020", "Adobe", "Adobe's applications", "Intel-based Macs", "80% performance improvement"]

{
  "propositions": [
    {
      "text": "Apple launched the M1 chip in 2020.",
      "entities": ["Apple", "M1 chip", "2020"]
    },
    {
      "text": "Adobe optimized their applications specifically for the M1 chip after its launch.",
      "entities": ["Adobe", "Adobe's applications", "M1 chip"]
    },
    {
      "text": "Adobe's applications running on the M1 chip improved performance by up to 80% compared to the same applications running on Intel-based Macs.",
      "entities": ["Adobe", "Adobe's applications", "M1 chip", "80% performance improvement", "Intel-based Macs"]
    }
  ],
  "relations": [
    {"src": 0, "dst": 1, "type": "causes", "symmetric": false},
    {"src": 1, "dst": 2, "type": "causes", "symmetric": false}
  ]
}

Example 2:
Passage: In September 2023, Apple replaced the Lightning connector with USB-C on the iPhone 15, after the European Union passed regulations requiring all mobile devices to use a standardized charging port.

Named entities: ["Apple", "Lightning connector", "USB-C connector", "iPhone 15", "European Union", "regulations", "standardized charging port", "mobile devices", "September 2023"]

{
  "propositions": [
    {
      "text": "The iPhone 15 uses a USB-C connector instead of the Lightning connector.",
      "entities": ["iPhone 15", "USB-C connector", "Lightning connector"]
    },
    {
      "text": "The European Union passed regulations requiring all mobile devices to use a standardized charging port.",
      "entities": ["European Union", "regulations", "mobile devices", "standardized charging port"]
    },
    {
      "text": "Apple changed from Lightning to USB-C on the iPhone 15 due to the European Union regulations in September 2023.",
      "entities": ["Apple", "Lightning connector", "USB-C connector", "iPhone 15", "European Union", "regulations", "September 2023"]
    }
  ],
  "relations": [
    {"src": 1, "dst": 2, "type": "causes", "symmetric": false},
    {"src": 2, "dst": 0, "type": "modifies", "symmetric": false}
  ]
}
"""

proposition_frame = """
Passage:
```
{passage}
```

Named entities: {named_entities}"""

prompt_template = [
    {"role": "system", "content": proposition_system},
    {"role": "user", "content": convert_format_to_template(
        original_string=proposition_frame,
        placeholder_mapping=None,
        static_values=None
    )}
]
