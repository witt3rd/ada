"""
main9_ada_personal_ai_assistant_v02.py - Advanced Digital Assistant (ADA) Personal AI Assistant

This script implements a voice-activated AI assistant named ADA (Advanced Digital Assistant).
It serves as a proof of concept for a personal AI assistant capable of performing various
tasks through voice commands and natural language processing.

Key Features:
1. Voice Activation: Listens for the activation keyword "Ada" to start processing commands.
2. Natural Language Processing: Utilizes advanced language models for understanding and
   responding to user queries.
3. Multi-Modal AI Integration: Incorporates multiple AI models including GPT-4 and Gemini 1.5
   for diverse task handling.
4. Text-to-Speech: Provides audible responses using ElevenLabs' text-to-speech technology.
5. Command Execution: Capable of running shell commands and interacting with the system.
6. Web Scraping: Can extract and process information from websites.
7. Code Generation: Assists in generating example code based on user requests.
8. Image Processing: Converts images to Vue.js components.
9. Configuration Management: Allows for customization of assistant settings.

Main Components:
- Workflow Functions: Implement various task-specific workflows (e.g., question answering,
  code generation, system configuration).
- Helper Functions: Provide utility functionalities like text processing and file handling.
- Audio I/O Functions: Manage audio recording, transcription, and text-to-speech conversion.
- Main Loop: Continuously listens for the activation keyword and processes user commands.

Dependencies:
- External Modules: tkinter, pydantic, sounddevice, elevenlabs, openai, google.generativeai
- Custom Modules: human_in_the_loop, llm, editor, parsers

Environment Setup:
- Requires API keys for OpenAI, Google Cloud, ElevenLabs, and Deepgram.
- Configuration is managed through a JSON file (config.json).

Usage:
1. Ensure all dependencies are installed and API keys are set in the environment.
2. Run the script to start the AI assistant.
3. Activate the assistant by saying "Ada" followed by a command or question.
4. The assistant will process the request and respond audibly.

Note: This is version 0.2.1 of the ADA AI Assistant, representing an early-stage
proof of concept. It demonstrates the integration of various AI technologies but
may require further refinement for production use.

Author: [Your Name]
Date: [Current Date]
Version: 0.2.1
"""
# ADA - Personal AI Assistant (v0.2.1)
# Proof of Concept

# CHANGES: (v0.2.1)
# - using textwrap.dedent() for better formatting of multi-line prompts
# - using deepgram instead of assembly ai for audio-to-text transcription
# - minor changes to the code generation prompts

import difflib
import json
import os
import subprocess
import sys
import wave
from datetime import datetime
from textwrap import dedent

import pyperclip
import requests
import sounddevice as sd
from bs4 import BeautifulSoup
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from markdownify import markdownify
from pydantic import BaseModel

from ada.modules import editor, human_in_the_loop, llm

load_dotenv()

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
if not ELEVENLABS_VOICE_ID:
    print(
        "❌ ElevenLabs voice ID not found. Please set the ELEVENLABS_VOICE_ID environment variable."
    )
    sys.exit()

ACTIVATION_KEYWORD = os.getenv("ACTIVATION_KEYWORD", "Ada")
PERSONAL_AI_ASSISTANT_NAME = os.getenv("PERSONAL_AI_ASSISTANT_NAME", "ADA")
CONFIG_FILE = os.getenv("CONFIG_FILE", "./config.json")
HUMAN_COMPANION_NAME = os.getenv("HUMAN_COMPANION_NAME", "Donald")

PERSONAL_AI_ASSISTANT_PROMPT_HEAD = dedent(f"""
    You are a friendly, ultra helpful, attentive, concise AI assistant named '{PERSONAL_AI_ASSISTANT_NAME}'.
    You work with your human companion '{HUMAN_COMPANION_NAME}' to build valuable experience through software.
    We both like short, concise, back-and-forth conversations.
""").strip()

try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        configuration = json.load(config_file)
except FileNotFoundError:
    configuration = {
        "working_directory": None,
    }
    # write
    with open(CONFIG_FILE, "w", encoding="utf-8") as config_file:
        json.dump(configuration, config_file, indent=2)

CHANNELS = 1
ITERATION_START_TIME = None

# --------------------- Agent Workflows ---------------------


def get_simple_keyword_ai_agent_router():
    """
    Decision making based on contents of prompt (Simple LLM Router).
    """
    return {
        # v0.2 New Flows w/two-way prompts
        "configure,configuration": configure_assistant_workflow,
        "example code": example_code_workflow,
        "view component": image_to_vue_component_workflow,
        #
        "bash,browser": run_bash_command_workflow,  # AI Agent | Agentic Workflow
        "shell": shell_command_workflow,  # AI Agent | Agentic Workflow
        "question": question_answer_workflow,  # AI Agent | Agentic Workflow
        "hello,hey,hi": soft_talk_workflow,  # AI Agent | Agentic Workflow
        #
        "exit": end_conversation_workflow,
    }


def image_to_vue_component_workflow(prompt: str):
    """
    Generate a Vue component from an image
    """

    class VueComponentResponse(BaseModel):
        vue_component: str

    class FileNameResponse(BaseModel):
        file_name: str

    speak(build_feedback_prompt("Select an image to generate a Vue component from."))

    open_file_path = human_in_the_loop.open_file()

    print(f"🎆 Image selected at {open_file_path}")

    if not open_file_path:
        speak(
            build_feedback_prompt("No image found in clipboard. Skipping this request.")
        )
        return

    speak(
        build_feedback_prompt(
            "Okay I see the image, Now I'll generate the Vue component based on the image and your request."
        )
    )

    component_response: VueComponentResponse = llm.gpt4t_w_vision_image_with_model(
        dedent("""
            You're a Senior Vue 3 developer. You build new Vue components using the Composition API with <script setup lang='ts'>.
            You strictly follow the REQUIREMENTS below.

            REQUIREMENTS:
            - Your current assignment is to build a new vue component that matches the image.
            - Return strictly the code for the Vue component including <template>, <script setup lang='ts'>, and <style> sections.
            - Use tailwind css to style the component.
            - Respond in this JSON format exclusively: {vue_component: ''}
        """).strip(),
        file_path=open_file_path,
        pydantic_model=VueComponentResponse,
    )

    file_name_response: FileNameResponse = llm.gpt4t_w_vision_json_prompt(
        dedent(f"""You're a Senior Vue 3 developer. You build new Vue components using the Composition API with <script setup lang='ts'>.
                   You've just created the VUE_COMPONENT. Now you're naming the component.

                   Create a concise and descriptive name for the component.
                   Respond in this JSON format exclusively: {{file_name: ''}}

                   VUE_COMPONENT:
                       {component_response.vue_component}
               """).strip(),
        pydantic_model=FileNameResponse,
    )

    # dump to .vue file
    file_path = os.path.join(
        configuration["working_directory"], file_name_response.file_name
    )

    # write
    with open(file_path, "w") as file:
        file.write(component_response.vue_component)

    speak(
        build_feedback_prompt(
            f"I've created the Vue component and named it {file_name_response.file_name}. Let me know if you want to make any edits."
        )
    )

    human_in_the_loop.open_file_in_editor_and_continue(file_path)

    requested_updates = human_in_the_loop.open_editor()

    if not requested_updates:
        speak(build_feedback_prompt("No changes requested. Component ready for use."))
        return

    component_to_update = component_response.vue_component

    update_component_response: VueComponentResponse = llm.gpt4t_w_vision_json_prompt(
        dedent(f"""You're a Senior Vue 3 developer. You build new Vue components using the Composition API with <script setup lang='ts'>.
                   You've just created the VUE_COMPONENT. A change from your product manager has come in and you're now tasked with updating the component.
                   You follow the REQUIREMENTS below to make sure the component is updated correctly.

                   REQUIREMENTS:
                     - Your current assignment is to make updates to the VUE_COMPONENT based on the changes requested by the product manager.
                     - Return strictly the code for the Vue component including <template>, <script setup lang='ts'>, and <style> sections.
                     - Use tailwind css to style the component.
                     - Respond in this JSON format exclusively: {{vue_component: ''}}

                   REQUESTED_CHANGES:
                      {requested_updates}

                   VUE_COMPONENT:
                      {component_to_update}
                """),
        pydantic_model=VueComponentResponse,
    )

    # write to file
    with open(file_path, "w") as file:
        file.write(update_component_response.vue_component)

    speak(
        build_feedback_prompt(
            "I've updated the Vue component based on your feedback. What's next?"
        )
    )

    pass


def run_bash_command_workflow(prompt: str):
    run_bash_prompt = dedent(
        f"""You are a friendly, ultra helpful, attentive, concise AI assistant named '{PERSONAL_AI_ASSISTANT_NAME}'.
                   You work with your human companion '{HUMAN_COMPANION_NAME}' to build valuable experience through software.

                   You've been asked to run the following bash COMMAND: '{prompt}'

                   Here are available bash COMMANDS you can run:

                   # chrome browser
                   browser() {{
                      open -a 'Google Chrome' $1
                   }}

                   # typescript playground
                   playt() {{
                      cursor "/Users/ravix/Documents/projects/experimental/playt"
                   }}

                   chats() {{
                      browser "https://aistudio.google.com/app/prompts/new_chat"
                      browser "https://console.anthropic.com/workbench"
                      browser "https://chat.openai.com/"
                   }}

                   Based on the COMMAND - RESPOND WITH THE COMMAND to run in this JSON format: {{bash_command_to_run: ''}}.

                   Exclude any new lines or code blocks from the command. Respond with exclusively JSON.

                   Your COMMAND will be immediately run and the output will be returned to the user.
                """
    )

    class BashCommandResponse(BaseModel):
        bash_command_to_run: str

    response: BashCommandResponse = llm.gpt4t_w_vision_json_prompt(
        run_bash_prompt, pydantic_model=BashCommandResponse
    )

    print("👧 Raw response: ", response)

    command = response.bash_command_to_run

    print(f"💻 {PERSONAL_AI_ASSISTANT_NAME} is running this command: ", command)
    try:
        command = "source ~/.bash_profile && " + command
        result = subprocess.run(
            command,
            shell=True,
        )
        print(f"💻 Command executed successfully: {command}")
        print(f"💻 Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"💻 Error executing command: {command}\n💻 Error: {e}")
        return

    soft_talk_prompt = dedent(
        f"""You are a friendly, ultra helpful, attentive, concise AI assistant named '{PERSONAL_AI_ASSISTANT_NAME}'.
                  You work with your human companion '{HUMAN_COMPANION_NAME}' to build valuable experience through software.

                  We both like short, concise, back-and-forth conversations.
                  We don't like small talk so we always steer our conversation back toward creating, building, product development, designing, and coding.

                  You've just helped your human companion run this bash COMMAND: {command}

                  Let your human companion know you've finished running the command and what you can do next.
               """
    )

    response = llm.gpro_1_5_prompt(soft_talk_prompt)

    speak(response)

    pass


def question_answer_workflow(prompt: str):
    question_answer_prompt = dedent(
        f"""{PERSONAL_AI_ASSISTANT_PROMPT_HEAD}
                  We don't like small talk so we always steer our conversation back toward creating, building, product development, designing, and coding.
                  We like to discuss in high level details without getting too technical.
                  Respond to the following question: {prompt}
               """
    )

    response = llm.gpro_1_5_prompt(question_answer_prompt)

    speak(response)

    pass


def soft_talk_workflow(prompt: str):
    soft_talk_prompt = dedent(
        f"""{PERSONAL_AI_ASSISTANT_PROMPT_HEAD}
                  We don't like small talk so we always steer our conversation back toward creating, building, product development, designing, and coding.
                  Respond to the following prompt: {prompt}
               """
    )

    response = llm.gpro_1_5_prompt(soft_talk_prompt)

    speak(response)

    return


def shell_command_workflow(prompt: str):
    shell_command_prompt = dedent(
        f"""You are a highly efficient, code-savvy AI assistant named '{PERSONAL_AI_ASSISTANT_NAME}'.
                  You work with your human companion '{HUMAN_COMPANION_NAME}' to build valuable experience through software.
                  Your task is to provide a JSON response with the following format: {{command_to_run: ''}} detailing the shell command
                  for the macOS bash shell to based on this question: {prompt}.

                  After generating the response, your command will be attached DIRECTLY to your human companions clipboard to be run.
               """
    )

    class ShellCommandModel(BaseModel):
        command_to_run: str

    response = llm.gpt4t_w_vision_json_prompt(
        prompt=shell_command_prompt,
        pydantic_model=ShellCommandModel,  # Assuming there's a suitable model or this parameter is handled appropriately within the function.
    )

    pyperclip.copy(response.command_to_run)

    completion_prompt = dedent(
        f"""You are a friendly, ultra helpful, attentive, concise AI assistant named '{PERSONAL_AI_ASSISTANT_NAME}'.
                  You work with your human companion '{HUMAN_COMPANION_NAME}' to build valuable experience through software.
                  We both like short, concise, back-and-forth conversations.

                  You've just attached the command '{response.command_to_run}' to your human companion's clipboard like they've requested.

                  Let your human companion know you've attached it and let them know you're ready for the  next task.
               """
    )

    completion_response = llm.gpro_1_5_prompt(completion_prompt)

    speak(completion_response)


def summarize_diff_workflow(start: str | dict, end: str | dict, file: str):
    """
    Summarize the diff between two strings
    """
    start = json.dumps(start, indent=2).splitlines()
    end = json.dumps(end, indent=2).splitlines()

    diff = difflib.unified_diff(start, end, fromfile="before", tofile="after")
    diffed = "\n".join(diff)

    summarize_prompt = dedent(f"""{PERSONAL_AI_ASSISTANT_PROMPT_HEAD}
                  Your companion has just finished editing the {file}.

                  You'll concisely summarize the changes made to the file in a 1 sentence summary.
                  The point is to communicate and acknowledge the changes made to the file.

                  The changes are:

                  {diffed}
               """)

    summarize_response = llm.gpro_1_5_prompt(summarize_prompt)

    speak(summarize_response)

    return diffed


def configure_assistant_workflow(prompt: str):
    """
    Configure settings for our assistant
    """

    configure_prompt = dedent(f"""{PERSONAL_AI_ASSISTANT_PROMPT_HEAD}
                  You've just opened a configuration file for your human companion.
                  Let your human companion know you've opened the file and are ready for them to edit it.
               """)

    prompt_response = llm.gpro_1_5_prompt(prompt=configure_prompt)

    speak(prompt_response)

    global configuration

    previous_configuration = configuration
    updated_config = human_file_json_prompt(configuration)
    with open(CONFIG_FILE, "w") as config_file:
        json.dump(updated_config, config_file, indent=2)

    summarize_diff_workflow(
        previous_configuration, updated_config, "configuration.json"
    )


def end_conversation_workflow(prompt: str):
    end_prompt = dedent(f"""{PERSONAL_AI_ASSISTANT_PROMPT_HEAD}
                  We're wrapping up our work for the day. You're a great engineering partner.
                  Thanks for all your help and for being a great engineering partner.

                  Respond to your human companions closing thoughts: {prompt}
               """)

    response = llm.gpro_1_5_prompt(end_prompt)

    speak(response)

    sys.exit()


def example_code_workflow(prompt: str):
    """
    Generate code for a given prompt
    """

    class ExampleCodeResponse(BaseModel):
        code: str

    class ExampleCodeFileNameResponse(BaseModel):
        file_name: str

    url_from_clipboard = pyperclip.paste()

    if not url_from_clipboard or "http" not in url_from_clipboard:
        speak(
            build_feedback_prompt(
                "I don't see a URL on your clipboard. Please paste a URL into your editor."
            )
        )

        url_from_clipboard = human_in_the_loop.open_editor()

    if not url_from_clipboard:
        speak(
            build_feedback_prompt(
                "Still no URL found in clipboard. Skipping this request."
            )
        )
        return

    print(f"🔗 Scraping URL found in clipboard: {url_from_clipboard}")

    speak(
        build_feedback_prompt(
            dedent("""I've found the URL in your clipboard.
                       I'll scrape the URL and example generate code for you.
                       But first, what about the example code would you like me to focus on?
                    """)
        )
    )

    feedback_for_code_generation = human_in_the_loop.open_editor()

    speak(
        build_feedback_prompt(
            f"Okay got it, I see you want to focus on '{feedback_for_code_generation}'. I'll generate the code for you now."
        )
    )

    scraped_markdown = scrape_to_markdown(url_from_clipboard)

    example_code_response_1: ExampleCodeResponse = llm.gpro_1_5_prompt_with_model(
        dedent(f"""You're a professional software developer advocate that takes pride in writing good code.
                   You take documentation, and convert it into runnable code.

                   You have a new request to generate code for the following url: '{url_from_clipboard}' with a focus on '{feedback_for_code_generation}'.

                   Given the scraped WEBSITE_CONTENT content below, generate working code to showcase how to run the code.

                   Focus on the code. Use detailed variable and function names. Comment every line of code explaining what it does.
                   Remember, this is code to showcase how the code works. It should be fully functional and runnable.
                   Respond in this JSON format exclusively: {{code: ''}}

                   WEBSITE_CONTENT:
                     {scraped_markdown}
                """),
        pydantic_model=ExampleCodeResponse,
    )

    print("👧 Raw response: v1\n\n", example_code_response_1.code)

    example_code_response_2 = llm.gpt4t_w_vision_json_prompt(
        dedent(f"""You are an elite level, principle software engineer.

                   You work with a co-engineer that likes to leave non-runnable code in the code so you're responsible for making sure it's runnable.
                   You've just generated the first draft EXAMPLE_CODE below.

                   You're now taking a second pass to clean it up to make sure it meets the REQUIREMENTS

                   REQUIREMENTS:
                      - Make sure it's immediately runnable and functional.
                      - Removing anything that isn't runnable code.
                      - This code will be immediately placed into a file and run.
                      - The code should be well commented so it's easy to understand.
                      - The code should be well formatted so it's easy to read.
                      - The code should use verbose variable and function names.
                      - You pay close attention to indentation.
                      - Respond in JSON format with the following keys: {{code: ''}}

                   EXAMPLE_CODE:
                     {example_code_response_1.code}
                """),
        pydantic_model=ExampleCodeResponse,
    )

    print("👧 Raw response: v2\n\n", example_code_response_2.code)

    example_code_response_3 = llm.gpt4t_w_vision_json_prompt(
        dedent(f"""You are a top-level programmer and super-expert in software engineering.

                   You've received a near final draft of code to finalize.
                   You work with a co-engineer that likes to leave non-runnable code in the code so you're responsible for making sure it's runnable.
                   You're taking a final pass to make sure the code is near perfect and fully runnable.
                   You follow the REQUIREMENTS below to make sure the code is top notch for production deployment.

                   REQUIREMENTS:
                      - Make sure the code is immediately runnable and functional.
                      - Removing anything that isn't runnable code.
                      - This code will be immediately placed into a file and run.
                      - The code follows expert coding best practices.
                      - The code should be well commented so it's easy to understand.
                      - The code should be well formatted so it's easy to read.
                      - The code should use verbose variable and function names.
                      - You pay close attention to indentation.
                      - Respond in JSON format with the following keys: {{code: ''}}

                   EXAMPLE_CODE:
                      {example_code_response_2.code}
                """),
        pydantic_model=ExampleCodeResponse,
    )

    print("👧 Raw response: v3\n\n", example_code_response_3.code)

    example_code_file_prompt = dedent(f"""{PERSONAL_AI_ASSISTANT_PROMPT_HEAD}

                  You've just generated the following CODE below for your human companion.
                  Create a file name for the code file that will be written to the following directory: {configuration['working_directory']}
                  The file name should be unique and descriptive of the code it contains.
                  Respond exclusively with the file name in the following JSON format: {{file_name: ''}}.

                  CODE:
                    {example_code_response_3.code}
               """)

    example_code_file_name_response = llm.gpt4t_w_vision_json_prompt(
        example_code_file_prompt,
        pydantic_model=ExampleCodeFileNameResponse,
    )

    new_file_name = example_code_file_name_response.file_name

    new_file_path = os.path.join((configuration["working_directory"]), new_file_name)

    # write the code to the file
    with open(new_file_path, "w") as file:
        file.write(example_code_response_3.code)

    print(f"✅ Code example written to {new_file_path}")

    speak(
        build_feedback_prompt(
            f"Code has been written to the working directory into a file named {new_file_name}. Let me know if you need anything else."
        )
    )

    pass


# --------------------- Helper Methods ---------------------


def human_file_json_prompt(contents: dict):
    """
    Prompt the user to edit the file
    """
    edited_contents = editor.edit(contents=json.dumps(contents, indent=2))
    edited_config = json.loads(edited_contents.decode())

    return edited_config


def scrape_to_markdown(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Convert the parsed HTML to Markdown using markdownify
    markdown = markdownify(str(soup), strip=["script", "style"])

    return markdown


def build_feedback_prompt(message: str):
    """
    Build a prompt using the existing prompt format and ask our assistant to respond given the 'message'
    """
    prompt = dedent(f"""{PERSONAL_AI_ASSISTANT_PROMPT_HEAD}
                        Concisely communicate the following message to your human companion: '{message}'
                     """)

    response = llm.gpro_1_5_prompt(prompt)

    return response


# --------------------- AUDIO I/O ---------------------


def speak(text: str):
    client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),  # Defaults to ELEVEN_API_KEY from .env
    )

    # text=text, voice="WejK3H1m7MI9CHnIjW9K",
    audio = client.generate(
        text=text,
        voice=ELEVENLABS_VOICE_ID,
        model="eleven_turbo_v2",
        # model="eleven_multilingual_v2",
    )

    play(audio)


def transcribe_audio_file(file_path):
    try:
        # STEP 1 Create a Deepgram client using the API key
        api_key = os.getenv("DEEPGRAM_API_KEY")
        dg_client = DeepgramClient(api_key)

        # STEP 2 Read the recorded audio file
        with open(file_path, "rb") as file:
            buffer_data = file.read()

        # STEP 2: Configure Deepgram options for audio analysis
        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-2", smart_format=True)

        # STEP 3: Call the transcribe_file method with the text payload and options
        response = dg_client.listen.prerecorded.v("1").transcribe_file(payload, options)

        # STEP 4: Await the response and extract the transcript
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]

        return transcript

    except Exception as e:
        print(f"Exception: {e}")
        return ""


def track_interaction_time():
    """Track the time it takes for the user to interact with the system in seconds."""
    global ITERATION_START_TIME
    if ITERATION_START_TIME:
        interaction_time = (datetime.now() - ITERATION_START_TIME).total_seconds()
        print(f"🕒 Interaction time: {interaction_time} seconds")
        ITERATION_START_TIME = None


def record_audio(duration=10, fs=44100):
    """Record audio from the microphone."""
    track_interaction_time()

    print("🔴 Recording...")
    recording = sd.rec(
        int(duration * fs), samplerate=fs, channels=CHANNELS, dtype="int16"
    )
    sd.wait()
    print("🎧 Recording Chunk Complete")
    global ITERATION_START_TIME
    ITERATION_START_TIME = datetime.now()
    return recording


def save_audio_file(recording, fs=44100, filename="output.wav"):
    """Save the recorded audio to a file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording)


def personal_ai_assistant_loop(
    audio_chunk_size=10, activation_keyword=ACTIVATION_KEYWORD, on_keywords=None
):
    while True:
        recording = record_audio(duration=audio_chunk_size)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}.wav"
        save_audio_file(recording, filename=filename)
        file_size = os.path.getsize(filename)
        print(f"📁 File {filename} has been saved with a size of {file_size} bytes.")
        transcript = transcribe_audio_file(filename)
        print("📝 transcript was:", transcript)
        if activation_keyword.lower() in transcript.lower():
            if on_keywords:
                on_keywords(transcript)
        os.remove(filename)


def text_after_keyword(transcript: str, keyword: str):
    """Extract and return the text that comes after a specified keyword in the transcript."""
    try:
        # Find the position of the keyword in the transcript
        keyword_position = transcript.lower().find(keyword.lower())
        if keyword_position == -1:
            # If the keyword is not found, return an empty string
            return ""
        # Extract the text after the keyword
        text_after = transcript[keyword_position + len(keyword) :].strip()
        return text_after
    except Exception as e:
        print(f"Error extracting text after keyword: {e}")
        return ""


def get_first_keyword_in_prompt(prompt: str):
    map_keywords_to_agents = get_simple_keyword_ai_agent_router()
    for keyword_group, agent in map_keywords_to_agents.items():
        keywords = keyword_group.split(",")
        for keyword in keywords:
            if keyword in prompt.lower():
                return agent, keyword
    return None, None


def on_activation_keyword_detected(transcript: str):
    print("✅ Activation keyword detected!, transcript is: ", transcript)

    prompt = text_after_keyword(transcript, ACTIVATION_KEYWORD)

    print("🔍 prompt is: ", prompt)

    agent_to_run, agent_keyword = get_first_keyword_in_prompt(prompt)

    if not agent_to_run:
        print("❌ No agent found for the given prompt.")
        return

    print(f"✅ Found agent via keyword '{agent_keyword}'")

    agent_to_run(prompt)


personal_ai_assistant_loop(on_keywords=on_activation_keyword_detected)
