# Explanation of the Chunking Mechanism

This document explains the 24MB file size limit and the pros and cons of the chunking mechanism used in the Class Recorder application.

### The Reason: Whisper API Limitation

The core reason for the file size limit is that the **OpenAI Whisper API does not accept files larger than 25 megabytes (MB)** for transcription.

Your `class_recorder/transcriber.py` file correctly identifies this and sets a slightly lower limit of 24MB as a safe margin:

```python
class WhisperTranscriber:
    # Whisper API limit: 25MB
    MAX_FILE_SIZE_MB = 24  # Stay slightly under limit
```

To handle lectures, which are often longer and result in files much larger than this, the application must split the audio into smaller pieces—or "chunks"—that the API can accept.

### Pros and Cons of this Approach

This chunking strategy is a common and necessary workaround, but it comes with its own set of advantages and disadvantages.

**Pros:**

1.  **Process Arbitrarily Long Audio:** The most significant advantage is that it allows the tool to transcribe lectures of any length, whether it's 30 minutes or 3 hours. Without chunking, any recording over the size limit would fail.
2.  **Increased Reliability:** If a network error occurs during one of the uploads, only that small chunk needs to be retried, not the entire large file. The code uses the `tenacity` library to handle this automatically.
3.  **Lower Memory Usage:** By processing the audio in smaller segments, the application avoids having to load a potentially very large file into memory all at once.

**Cons:**

1.  **Potential for Context Loss:** The Whisper model's accuracy is highest when it has more context. When a file is split, the model loses context at the beginning and end of each chunk. This can sometimes result in lower transcription accuracy for words or phrases that occur right at the split.
2.  **Increased Complexity:** The code has to manage the entire chunking workflow: splitting the file with `ffmpeg`, sending each chunk for transcription, and then carefully stitching the text and timestamps back together in the correct order. This makes the `transcriber.py` module more complex.
3.  **Timestamp Recalculation:** Timestamps from each chunk are relative to the start of that chunk (e.g., they start from `00:00:00`). The application must manually add an offset to each segment's timestamp to ensure they are accurate in the final, combined transcript. An error in this logic would lead to incorrect timing.

In summary, the chunking mechanism is a necessary and practical solution to a hard API limit, but it introduces complexity and potential (though usually minor) transcription inaccuracies at the points where the audio is split.
