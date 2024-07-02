"""
llm.py - Language Model Integration Module for ADA AI Assistant

This module provides a unified interface for interacting with various large language models (LLMs)
and AI services, specifically tailored for the ADA AI Assistant project. It encapsulates the
complexity of working with different AI providers and models, offering a consistent API for
text generation, image analysis, and structured data extraction.

Key Features:
1. Integration with multiple AI services:
   - Google's Generative AI (Gemini 1.5)
   - OpenAI's GPT-4 and GPT-4 Vision
2. Support for text-only and multimodal (text + image) prompts
3. Structured output parsing using Pydantic models
4. Image encoding for vision-based AI models
5. Environment variable management for API keys

Main Functions:
- gpro_1_5_prompt: Generate text using Google's Gemini 1.5 Pro model
- gpro_1_5_prompt_with_model: Generate structured output using Gemini 1.5 Pro with Pydantic models
- gpt4t_w_vision_json_prompt: Generate JSON-structured responses using GPT-4 Turbo
- gpt4t_w_vision: Generate free-form text responses using GPT-4 Turbo
- gpt4t_w_vision_image_with_model: Analyze images and generate structured responses using GPT-4 Vision

Utility Functions:
- encode_image: Convert image files to base64 encoded strings for API requests

Dependencies:
- google.generativeai: For interacting with Google's Generative AI models
- openai: For interacting with OpenAI's models
- pydantic: For data validation and settings management
- dotenv: For loading environment variables
- base64: For encoding images

Usage:
    from modules import llm

    # Text generation with Gemini 1.5
    response = llm.gpro_1_5_prompt("Tell me about AI assistants")

    # Structured output with GPT-4
    class MyModel(BaseModel):
        field1: str
        field2: int

    result = llm.gpt4t_w_vision_json_prompt("Analyze this data", pydantic_model=MyModel)

    # Image analysis with GPT-4 Vision
    image_analysis = llm.gpt4t_w_vision_image_with_model("Describe this image", "path/to/image.jpg", pydantic_model=MyModel)

Note:
This module requires valid API keys for Google Cloud and OpenAI to be set in the environment
variables GOOGLE_API_KEY and OPENAI_API_KEY respectively. These can be loaded from a .env file
or set in the system environment.

The module is designed to be easily extendable to support additional AI models and services
as needed for the ADA AI Assistant project.
"""

import base64
import os

import google.generativeai as genai
import openai
from dotenv import load_dotenv
from pydantic import BaseModel

from ada.modules import parsers

# Load environment variables from .env file
load_dotenv()

GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if GOOGLE_GENAI_API_KEY is None:
    raise ValueError("Google GenAI API key not found in environment variables")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found in environment variables")

#

genai.configure(api_key=GOOGLE_GENAI_API_KEY)
openai.api_key = OPENAI_API_KEY

#


def gpro_1_5_prompt(prompt) -> str:
    """
    Generates content based on the provided prompt using the Gemini 1.5 API model and returns the text part of the first candidate's content.

    Args:
    - prompt (str): The prompt to generate content for.

    Returns:
    - str: The text part of the first candidate's content from the generated response.
    """
    model_name = "models/gemini-1.5-pro-latest"
    gen_config = genai.GenerationConfig()
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt, request_options={})
    return response.candidates[0].content.parts[0].text


def gpro_1_5_prompt_with_model(prompt, pydantic_model: BaseModel) -> BaseModel:
    """
    Generates content based on the provided prompt using the Gemini 1.5 API model and returns the text part of the first candidate's content.

    Args:
    - prompt (str): The prompt to generate content for.

    Returns:
    - str: The text part of the first candidate's content from the generated response.
    """
    model_name = "models/gemini-1.5-pro-latest"
    gen_config = genai.GenerationConfig()
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt, request_options={})
    response_text = response.candidates[0].content.parts[0].text
    if "```json" in response_text:
        return pydantic_model.model_validate(
            parsers.parse_json_from_gemini(response_text)
        )
    else:
        return pydantic_model.model_validate_json(response_text)


def gpt4t_w_vision_json_prompt(
    prompt: str,
    model: str = "gpt-4-turbo-2024-04-09",
    instructions: str = "You are a helpful assistant that response in JSON format.",
    pydantic_model: BaseModel = None,
) -> str:
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,  # Added instructions as a system message
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    response_text = response.choices[0].message.content
    print(f"Text LLM response: {response_text}")

    as_model = pydantic_model.model_validate_json(response_text)

    return as_model


def gpt4t_w_vision(
    prompt: str,
    model: str = "gpt-4-turbo-2024-04-09",
    instructions: str = "You are a helpful assistant.",
) -> str:
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,  # Added instructions as a system message
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    response_text = response.choices[0].message.content
    return response_text


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt4t_w_vision_image_with_model(
    prompt: str,
    file_path: str,
    model: str = "gpt-4-turbo-2024-04-09",
    instructions: str = "You are a helpful assistant that specializes in image analysis.",
    pydantic_model: BaseModel = None,
):
    file_extension = file_path.split(".")[-1]

    base64_image = encode_image(file_path)

    print("base64_image", base64_image)

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file_extension};base64,{base64_image}"
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )

    print("response", response)

    response_text = response.choices[0].message.content

    print("response_text", response_text)

    parsed_response = pydantic_model.model_validate_json(response_text)

    return parsed_response
