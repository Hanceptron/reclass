# LLM Testing Guide

Here are the instructions to test the LLM output with a downloaded audio file.

## Audio Format

You can use `.mp3` files directly. The Whisper API supports various formats including `mp3`, `m4a`, `wav`, and more, so no conversion is necessary.

## Processing Workflow

1.  **Create a Test Directory:** Inside the `recordings` folder, create a new directory for your test. For example:
    ```
    recordings/LLMTest/
    ```

2.  **Add Your Audio File:** Place your downloaded `.mp3` audio file into the new directory:
    ```
    recordings/LLMTest/my_test_lecture.mp3
    ```

3.  **Run the `process` Command:** Use the following terminal command to start the transcription and summarization process:
    ```bash
    recorder process recordings/LLMTest/my_test_lecture.mp3
    ```

This command will automatically:
- Transcribe the audio file using the Whisper API.
- Generate the summary using the configured OpenRouter LLM.
- Save the output files (`.txt`, `.json`, and `.md`) into the `recordings/LLMTest/` directory.
