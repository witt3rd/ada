"""
parser.py - JSON Parsing Utility for ADA AI Assistant

This module provides utility functions for parsing JSON-like strings, particularly
those generated by AI models like Gemini. It's designed to handle and clean up
JSON responses that may include additional formatting or markdown-style code blocks.

Key Features:
1. JSON Extraction: Capable of extracting JSON content from within markdown-style
   code blocks (```json ... ```) using regular expressions.
2. Flexible Parsing: Handles both clean JSON strings and those embedded within
   other text or formatting.
3. Error Handling: Gracefully handles parsing errors, returning None instead of
   raising exceptions for invalid JSON.

Main Function:
- parse_json_from_gemini(json_str: str) -> dict | None:
    Parses a JSON-like string, potentially embedded in markdown-style code blocks.
    Returns a dictionary representation of the JSON or None if parsing fails.

    Args:
        json_str (str): A string containing JSON data, possibly within code blocks.

    Returns:
        dict | None: A dictionary representing the parsed JSON object, or None if
                     parsing fails.

Usage Example:
    from parser import parse_json_from_gemini

    json_response = '''
    ```json
    {
      "key1": "value1",
      "key2": "value2"
    }
    ```
    '''
    parsed_data = parse_json_from_gemini(json_response)
    if parsed_data:
        print(parsed_data)
    else:
        print("Failed to parse JSON")

Dependencies:
- json: For JSON parsing functionality.
- re: For regular expression operations to extract JSON from code blocks.

Note:
This module is particularly useful when working with AI models that may return
JSON data within formatted text responses. It's designed to be robust against
various formatting inconsistencies that might occur in such responses.

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

import json
import re


def parse_json_from_gemini(json_str: str):
    """Parses a dictionary from a JSON-like object string.

    Args:
      json_str: A string representing a JSON-like object, e.g.:
        ```json
        {
          "key1": "value1",
          "key2": "value2"
        }
        ```

    Returns:
      A dictionary representing the parsed object, or None if parsing fails.
    """

    try:
        # Remove potential leading/trailing whitespace
        json_str = json_str.strip()

        # Extract JSON content from triple backticks and "json" language specifier
        json_match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)

        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return None