# Guide to Implementing Overlapping Chunks

This guide explains how to modify your script to use overlapping audio chunks for better transcription accuracy.

### The Advantage of Overlap

Creating an overlap between chunks (e.g., 1 minute) provides the Whisper model with more context at the start and end of each segment. This can significantly improve transcription accuracy at the chunk boundaries, which is a common source of errors.

Implementing this is a two-part process:
1.  **Modify the `ffmpeg` command** to create overlapping audio chunks.
2.  **Handle the overlap** by intelligently merging the transcription results to avoid duplicating text.

---

### 1. Creating Overlapping Chunks

You need to modify the `_split_audio` method in `class_recorder/transcriber.py`. Replace the existing method with this new version, which creates chunks with a specified overlap.

```python
# In class_recorder/transcriber.py

def _split_audio(self, audio_path, temp_dir, chunk_duration_min, overlap_min=1):
    """Split audio into overlapping chunks using ffmpeg."""
    chunks = []
    chunk_duration_sec = chunk_duration_min * 60
    overlap_sec = overlap_min * 60
    
    # Get total duration of the audio file
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    total_duration = float(result.stdout.strip())
    
    # Calculate the start time for each chunk
    start_time = 0
    i = 0
    while start_time < total_duration:
        chunk_path = temp_dir / f"chunk_{i:03d}.m4a"
        
        split_cmd = [
            'ffmpeg', '-i', str(audio_path),
            '-ss', str(start_time),
            '-t', str(chunk_duration_sec),
            '-c', 'copy',
            '-y',
            str(chunk_path)
        ]
        subprocess.run(split_cmd, capture_output=True, check=True)
        chunks.append(chunk_path)
        
        # Move the start time for the next chunk
        start_time += chunk_duration_sec - overlap_sec
        i += 1

    return chunks
```

Next, you need to update the call to this function inside the `_transcribe_chunked` method to enable the overlap.

```python
# In class_recorder/transcriber.py, inside the _transcribe_chunked method...

# ...
chunk_duration_min = 30
overlap_min = 1  # Define the overlap in minutes

# Split file using ffmpeg with overlap
chunks = self._split_audio(audio_path, temp_dir, chunk_duration_min, overlap_min)
# ...
```

---

### 2. The Challenge: Merging the Transcripts

This is the more complex part of the task. After implementing the change above, your script will transcribe overlapping audio, which means you will get duplicate text and segments in the results.

You need to "stitch" the results together intelligently.

A robust way to do this involves using the `start` and `end` timestamps of the transcribed segments that you get from Whisper's `verbose_json` response.

The high-level logic for updating the `_transcribe_chunked` method is as follows:

1.  **Transcribe the first chunk** and add all of its segments to your final list of segments.
2.  **Keep track of the "true" end time** of the last segment you've added to the final list.
3.  **For each subsequent chunk:**
    a. Transcribe the chunk.
    b. Before adding its segments to the final list, you must identify where the new, non-overlapping information begins.
    c. Discard all segments from this new chunk whose start times fall within the already-transcribed portion of the previous chunk.
    d. Add the remaining, truly "new" segments to your final list, making sure to adjust their timestamps to be relative to the global recording time.

This is a non-trivial logic change and will require careful state management as you loop through the chunks. I recommend starting with the `_split_audio` change and then examining the raw output to guide your implementation of the merging logic.
