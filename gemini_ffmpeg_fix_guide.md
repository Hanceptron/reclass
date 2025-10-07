# Guide to Fixing the FFmpeg `CalledProcessError`

This guide explains the cause of the `RetryError` and provides the fix for the `ffmpeg` command in your `transcriber.py` script.

### Error Analysis

The error `RetryError[<...CalledProcessError>]` indicates that the audio splitting process is failing, and the system is retrying it multiple times before giving up. The root cause is a `CalledProcessError` from the `ffmpeg` command used to split the audio chunks.

This happens because the command uses the argument `-c copy`, which tells `ffmpeg` to copy the audio stream directly without re-encoding. This fails when your input file is a `.mp3` and you are trying to create `.m4a` chunks, as the internal audio formats are incompatible for a direct copy.

### The Solution

To fix this, you must remove the `-c copy` argument to allow `ffmpeg` to re-encode the audio into the correct format for the `.m4a` container (AAC).

1.  **Open the file:** `class_recorder/transcriber.py`

2.  **Locate the `_split_audio` method.**

3.  **Modify the `split_cmd` list:** Find the `split_cmd` list inside the `while` loop and either delete or comment out the line `'-c', 'copy',`.

**Your corrected `split_cmd` should look like this:**

```python
# Inside the _split_audio method of WhisperTranscriber

# ... inside the while loop ...
split_cmd = [
    'ffmpeg', '-i', str(audio_path),
    '-ss', str(start_time),
    '-t', str(chunk_duration_sec),
    # '-c', 'copy',  <-- This line should be removed or commented out
    '-y',
    str(chunk_path)
]
# ...
```

After making this change, save the file and run your `recorder process` command again. The `ffmpeg` error will be resolved, and the transcription process should now complete successfully.
