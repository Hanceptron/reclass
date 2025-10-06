# Gemini Log

## Project Analysis

**Project:** Class Recorder

**Objective:** A command-line tool to record, transcribe, and summarize lectures, with output formatted for Obsidian.

**Core Technologies:**
*   Python
*   Click (for CLI)
*   sounddevice (for audio recording)
*   OpenAI Whisper API (for transcription)
*   OpenRouter (for summarization)

**Dependencies:**
*   `sounddevice`: Audio recording.
*   `soundfile`: Reading/writing audio files.
*   `numpy`, `scipy`: Numerical operations for audio.
*   `openai`: Interacting with the Whisper API.
*   `click`: Creating the command-line interface.
*   `python-dotenv`: Loading API keys from `.env` file.
*   `pyyaml`: Loading configuration from `config.yaml`.
*   `tenacity`: For retrying API calls.

**Workflow:**
1.  Record audio via `recorder record <course_folder>`.
2.  Transcribe audio using Whisper.
3.  Summarize transcript using an LLM via OpenRouter.
4.  Save audio, transcript, and summary to the specified course folder.

**File Structure:**
*   `class_recorder/`: Main application package.
*   `recordings/`: Output directory for course notes (Obsidian vault).
*   `temp/`: Temporary file storage for chunking large audio files.
*   `config.yaml`: Configuration for API models, recording settings, etc.
*   `.env`: (Expected) for API keys.
*   `requirements.txt`: Python dependencies.

**Initial Observations:**
*   `project.md` is currently empty.
*   `startup.md` appears to be a quick-start guide, similar to the `README.md`.
*   The project is well-documented with a clear `README.md`.

**Next Steps:**
*   Analyze the source code in the `class_recorder/` directory to understand the implementation details.