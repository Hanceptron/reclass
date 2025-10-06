# Command Test List

Use this numbered list of commands to exercise every major feature of Class Recorder.

1. **Set up environment**
   ```bash
   cd ~/Projects/class-recorder
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```
2. **Configure keys and folders**
   ```bash
   cp .env.example .env  # if you created an example; otherwise edit .env directly
   open -a "TextEdit" .env
   open config.yaml
   mkdir -p recordings/Math101
   mkdir -p temp
   ```
3. **List available microphones**
   ```bash
   recorder devices
   ```
4. **Record, transcribe, and summarize**
   ```bash
   recorder record recordings/Math101
   ```
5. **Record with a specific input device** (replace `2` with the ID you saw in step 3)
   ```bash
   recorder record recordings/Math101 --device 2
   ```
6. **Process an existing audio file**
   ```bash
   recorder process recordings/Math101/2025-10-04-Derivatives.m4a
   ```
7. **Transcribe only**
   ```bash
   recorder transcribe-only recordings/Math101/2025-10-04-Derivatives.m4a
   ```
8. **Summarize only**
   ```bash
   recorder summarize-only recordings/Math101/2025-10-04-Derivatives.txt
   ```
9. **Deactivate the virtual environment when finished**
   ```bash
   deactivate
   ```
