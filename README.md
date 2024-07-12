# ADA - Personal AI Assistant (v0.2.1)

ADA (Advanced Digital Assistant) is a proof-of-concept personal AI assistant that uses voice recognition, natural language processing, and various AI models to help with tasks, answer questions, and interact with your computer.

Original: [Dan Isler's gist](https://gist.github.com/disler/1d926e312b2f46474b1773bace21f014)

## Features

- Voice activation with the keyword "Ada"
- Natural language interaction
- Integration with multiple AI models (GPT-4, Gemini 1.5)
- Text-to-speech responses using ElevenLabs
- Ability to run shell commands
- Web scraping and code generation
- Configuration management
- Image-to-Vue component generation

## Components

1. **main9_ada_personal_ai_assistant_v02.py**: The main script that runs the AI assistant loop and manages interactions.
2. **llm.py**: Handles interactions with various language models (GPT-4, Gemini 1.5).
3. **editor.py**: Provides functionality for editing files.
4. **human_in_the_loop.py**: Manages human interaction in the AI workflow.
5. **parsers.py**: Contains utility functions for parsing JSON responses.
6. **voice_recorder.py**: Handles continuous voice recording and transcription.

## Setup

1. Clone the repository:

   ```sh
   git clone git@github.com:witt3rd/ada.git
   cd ada
   ```

2. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:

   ```sh
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ELEVEN_API_KEY=your_elevenlabs_api_key
   DEEPGRAM_API_KEY=your_deepgram_api_key
   ```

4. Run the main script:

   ```sh
   python main.py
   ```

## Usage

1. Start the assistant by running the main script.
2. Activate the assistant by saying "Ada" followed by your command or question.
3. The assistant will process your request and respond using text-to-speech.

## Available Commands

- Configure settings: "Ada, configure the assistant"
- Generate example code: "Ada, give me an example code for..."
- Create a Vue component from an image: "Ada, create a view component for this image"
- Run shell commands: "Ada, run this bash command..."
- Ask questions: "Ada, what is..."
- End the conversation: "Ada, exit"

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
