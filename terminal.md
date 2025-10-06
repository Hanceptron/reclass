# In-Class Quick Terminal Checklist

Stay low-key and start recording fast. Follow the commands in order.

1. **Open Terminal** (Spotlight → type `Terminal` → Enter).
2. **Navigate to project folder**
   ```bash
   cd ~/Projects/class-recorder
   ```
3. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```
4. **Start recording immediately** (replace `Math101` with the course folder you set up)
   ```bash
   recorder record recordings/Math101
   ```
   - Optional mic override (if you know the ID from `recorder devices`):
     ```bash
     recorder record recordings/Math101 --device 2
     ```
5. **Let it run quietly**. Terminal shows the elapsed timer; keep the window minimized or dim the screen.
6. **Finish the lecture**
   - Press `Ctrl + C` once to stop.
   - Type a short lecture title when prompted (e.g., `AI search`) and hit Enter.
   - When asked, press Enter to confirm transcription/summarization (or `n` if you’ll do it later).
7. **After processing completes**
   - Check `recordings/Math101/` for `.m4a`, `.txt`, `.md`, and `_timestamps.json` files.
   - To leave the venv when you’re done for the day:
     ```bash
     deactivate
     ```

**Pro tip:** Before class, run `recorder devices` once at home to memorize your mic ID so you can skip that step in the room.
