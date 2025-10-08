# Class Recorder Command Reference

Here is a complete list of all the commands available in the Class Recorder tool, along with their arguments and descriptions.

---

### Main Workflow Commands

#### `recorder record <course_folder>`
This is the primary command. It records audio from your default microphone, and then automatically initiates the transcription and summarization process.

*   **`<course_folder>`**: (Required) The path to the directory where the output files will be saved (e.g., `recordings/Math101`).
*   **`--device <ID>`** or **`-d <ID>`**: (Optional) The numerical ID of a specific microphone you want to use. Use `recorder devices` to see the available IDs.

**Example:**
```bash
recorder record recordings/CS101 --device 2
```

#### `recorder process <audio_file>`
This command processes an existing audio file. It will run both the transcription and summarization steps on the given file.

*   **`<audio_file>`**: (Required) The path to the audio file you want to process (e.g., `recordings/CS101/lecture.m4a`).

**Example:**
```bash
recorder process recordings/CS101/2025-10-07-lecture.m4a
```

---

### Partial Workflow Commands

#### `recorder summarize-only <transcript_file>`
Use this command when you already have a text transcript and only want to run the LLM summarization step.

*   **`<transcript_file>`**: (Required) The path to the `.txt` file containing the transcript.

**Example:**
```bash
recorder summarize-only recordings/CS101/2025-10-07-lecture.txt
```

#### `recorder transcribe-only <audio_file>`
Use this command if you only want to transcribe an audio file and do not need an LLM summary. This will generate the `.txt` and `_timestamps.json` files.

*   **`<audio_file>`**: (Required) The path to the audio file you want to transcribe.

**Example:**
```bash
recorder transcribe-only recordings/CS101/2025-10-07-lecture.m4a
```

---

### Utility Command

#### `recorder devices`
This command lists all the audio input devices (microphones) connected to your system that `sounddevice` can detect. It displays their names and device IDs, which can be used with the `record` command's `--device` option.

**Example:**
```bash
recorder devices
```
