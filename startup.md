# Class Recorder Quick Start

If you can open a terminal and copy/paste, you can use Class Recorder. Follow these steps in order and you will end up with recorded lectures, transcripts, and Obsidian-ready notes.

## 1. One-Time Computer Setup

1. **Install Homebrew** (skip if already installed): visit https://brew.sh and follow the two-line command.
2. **Install ffmpeg**: open Terminal and run `brew install ffmpeg`. When it finishes, type `ffmpeg -version` to make sure it worked.
3. **Create a project folder**: choose where you want the app to live, then run:
   ```bash
   mkdir -p ~/Projects/class-recorder
   cd ~/Projects/class-recorder
   ```
4. **Download this project**: copy the project files into that folder (use `git clone ...` or copy from Finder if you already have the files).
5. **Create a Python virtual environment** (keeps dependencies clean):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
6. **Install the Python package**:
   ```bash
   pip install -e .
   ```
7. **Add your API keys**:
   - Open the `.env` file in any text editor.
   - Replace `sk-your-openai-key` and `sk-or-your-openrouter-key` with your real keys from OpenAI and OpenRouter.
   - Save the file.
8. **Review `config.yaml`**: the defaults work for most people. Only change them if you know why.

## 2. Prepare Your Obsidian Vault Folder

1. Open the `recordings/` folder in Finder.
2. Create a subfolder for each course (example: `Math101`, `Physics202`).
3. (Optional) Point Obsidian to this `recordings/` folder so new notes appear automatically.

## 3. Grant Microphone Permission (macOS)

1. The first time you record, macOS will ask for permission.
2. If you miss the prompt: System Settings → Privacy & Security → Microphone → enable the terminal or VS Code you use.

## 4. Everyday Workflow

1. **Activate the virtual environment** whenever you open a new terminal window:
   ```bash
   cd ~/Projects/class-recorder
   source venv/bin/activate
   ```
2. **List microphones** (optional, helps pick the right device):
   ```bash
   recorder devices
   ```
   - Look at the numbers in brackets. The default mic shows `[DEFAULT]`.
3. **Record a lecture**:
   ```bash
   recorder record recordings/Math101
   ```
   - Replace `Math101` with the folder for your class.
   - Speak normally; you’ll see "Recording..." in the terminal.
   - Press `Ctrl + C` when the lecture ends.
   - When asked for a class name, type something short like `Derivatives` and press Enter.
4. **Automatic processing**:
   - The app converts the audio to `.m4a`.
   - It then sends the audio to Whisper for transcription.
   - Finally, it asks the LLM for an Obsidian-formatted summary.
   - You’ll see progress messages in the terminal. This can take a few minutes depending on lecture length.
5. **Check your files** in the course folder:
   - `YYYY-MM-DD-Topic.m4a` (audio)
   - `YYYY-MM-DD-Topic.txt` (transcript)
   - `YYYY-MM-DD-Topic_timestamps.json` (detailed timings)
   - `YYYY-MM-DD-Topic.md` (summary ready for Obsidian)

## 5. Processing Existing Files

- If you recorded audio some other way, drop the `.m4a` into the course folder and run:
  ```bash
  recorder process recordings/Math101/2025-10-04-Derivatives.m4a
  ```
- Want only the transcript? Use `recorder transcribe-only <file>`.
- Already have a transcript but need the summary? Use `recorder summarize-only <file>`.

## 6. Troubleshooting Cheatsheet

- **"Module not found" or command not recognized**: virtual environment is probably inactive; run `source venv/bin/activate`.
- **`ffmpeg` not found**: reinstall with `brew install ffmpeg`.
- **API errors**: double-check the keys in `.env`, then reopen the terminal so the app reloads them.
- **Permission denied for microphone**: revisit System Settings → Privacy & Security → Microphone and enable your terminal.
- **Huge audio files**: the app splits anything bigger than 24 MB automatically. No action needed.

## 7. Shutting Down Cleanly

- When you finish recording and processing, press `Ctrl + D` or close the terminal window.
- To deactivate the virtual environment in the same session, run `deactivate`.

## 8. Quick Recap

1. Activate venv → `source venv/bin/activate`
2. Record → `recorder record recordings/YourCourse`
3. Press `Ctrl + C`, name the lecture, wait for processing
4. Open the `.md` note in Obsidian and review your summary

That’s it—repeat for each class and you’ll build a searchable library of lecture recordings, transcripts, and summaries with almost no manual effort.
