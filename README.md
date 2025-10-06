# Class Recorder

Record, transcribe, and summarize lectures with a single command. This project pairs local audio capture with OpenAI Whisper API transcription and OpenRouter LLM summarization, producing Obsidian-ready notes.

## Project Structure

```
class-recorder/
├── class_recorder/
│   ├── __init__.py
│   ├── cli.py              # Click commands (main entry point)
│   ├── recorder.py         # Audio recording with sounddevice
│   ├── transcriber.py      # Whisper API integration
│   ├── summarizer.py       # OpenRouter LLM integration
│   ├── config.py           # Configuration management
│   └── utils.py            # Helper functions
│
├── recordings/             # Obsidian vault folder (create course folders)
├── temp/                   # Temporary workspace for recordings/chunks
├── .env                    # API keys (gitignored, filled by you)
├── config.yaml             # Runtime settings
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

1. **Install system dependencies**

   ```bash
   brew install ffmpeg
   ffmpeg -version
   ```

2. **Create virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install package in editable mode**

   ```bash
   pip install -e .
   ```

4. **Configure environment**

   - Copy `.env` and add your own keys.
   - Adjust `config.yaml` if you prefer different models or recording parameters.
   - Create course folders under `recordings/` and ensure `temp/` exists.

## Usage

### Basic workflow

```bash
# List available microphones
recorder devices

# Record, transcribe, and summarize in one shot
recorder record recordings/Math101
```

During recording:
- Press `Ctrl+C` to stop
- Provide the lecture title when prompted

The following files are generated inside the course folder:
- `.m4a` audio file
- `.txt` transcript
- `_timestamps.json` metadata with segment timing
- `.md` summary ready for Obsidian

### Advanced commands

```bash
# Process an existing recording
recorder process recordings/Math101/2025-10-04-Derivatives.m4a

# Transcribe only (no summary)
recorder transcribe-only recordings/Math101/2025-10-04-Derivatives.m4a

# Summarize an existing transcript
recorder summarize-only recordings/Math101/2025-10-04-Derivatives.txt

# Specify an input device by ID
recorder record recordings/Math101 --device 2
```

## Large lecture handling

Whisper accepts files up to 25 MB. Anything larger triggers automatic chunking:

1. Split into 30-minute segments via `ffmpeg`
2. Transcribe each segment with Whisper
3. Adjust timestamps and merge into a single transcript
4. Remove temporary chunk files

## Cost considerations (approximate)

| Service             | Rate                 | 30 Lectures (1 hr) |
|---------------------|----------------------|--------------------|
| Whisper API         | $0.006 per minute    | $10.80             |
| Gemini 2.5 Flash    | $0.003 per summary   | $0.09              |
| **Total per term**  |                      | **~$11**           |

Reduce cost by trimming silence, lowering bitrate to 64 kbps, or batching processing during off-peak hours.

## Obsidian integration

- Point Obsidian to the `recordings/` directory, or
- Symlink an existing vault: `ln -s /path/to/Obsidian/Lectures recordings`

Output summaries include YAML frontmatter, callouts, timestamps, tables, and optional Mermaid diagrams based on the LLM response.

## Troubleshooting

- **Microphone permission denied:** enable the terminal application under System Settings → Privacy & Security → Microphone.
- **`ffmpeg` not found:** install via Homebrew and ensure it is on your PATH.
- **File size errors:** lower `recording.bitrate` in `config.yaml` or trim audio.
- **API rate limiting:** add short delays between chunk uploads in `transcriber.py` if necessary.

## Development

Run formatting and static analysis as needed, e.g. `ruff` or `black`, and keep API keys out of version control. Contributions are welcome—open issues or PRs for enhancements like custom prompts, batch scripts, or GUI wrappers.

