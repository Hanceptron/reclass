# Guide to Merging Overlapping Transcripts

This guide provides the updated `_transcribe_chunked` method for `class_recorder/transcriber.py`. This new version correctly merges the transcripts from the overlapping audio chunks you created.

---

### Updated `_transcribe_chunked` Method

Replace your existing `_transcribe_chunked` method with the code below. This function contains the complete logic for splitting, transcribing, and merging the overlapping segments.

```python
# In class_recorder/transcriber.py

def _transcribe_chunked(self, audio_path, course_folder, base_name):
    """Split large files, transcribe overlapping chunks, and merge them."""
    temp_dir = Path(config.get('storage.temp_dir', './temp'))
    temp_dir.mkdir(exist_ok=True)
    
    chunk_duration_min = 30
    overlap_min = 1  # Using a 1-minute overlap
    overlap_sec = overlap_min * 60
    
    # Split file into overlapping chunks
    chunks = self._split_audio(audio_path, temp_dir, chunk_duration_min, overlap_min)
    
    print(f"\n\ud83d\udcca Split into {len(chunks)} overlapping chunks")
    
    all_segments = []
    last_global_end_time = 0.0
    chunk_start_offset = 0.0

    for i, chunk in enumerate(chunks):
        print(f"\n\ud83d\udd04 Transcribing chunk {i + 1}/{len(chunks)}...")
        
        with open(chunk, 'rb') as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format=self.response_format,
                language=self.language,
                temperature=self.temperature
            )
        
        current_segments = self._extract_segments(response)

        if not current_segments:
            # If a chunk has no segments, skip it
            chunk.unlink()
            chunk_start_offset += (chunk_duration_min * 60) - overlap_sec
            continue

        for segment in current_segments:
            segment_global_start = chunk_start_offset + segment['start']
            segment_global_end = chunk_start_offset + segment['end']

            # Only add segments that start after the last known global end time
            if segment_global_start >= last_global_end_time:
                # Adjust segment timestamps to be global
                segment['start'] = segment_global_start
                segment['end'] = segment_global_end
                
                all_segments.append(segment)
                last_global_end_time = segment_global_end
        
        # Update the offset for the start of the next chunk
        chunk_start_offset += (chunk_duration_min * 60) - overlap_sec
        
        # Clean up the processed chunk
        chunk.unlink()
    
    # Reconstruct the full text from the merged segments
    full_text = ' '.join([s['text'].strip() for s in all_segments])
    total_duration = all_segments[-1]['end'] if all_segments else 0.0
    
    # Save combined transcript
    transcript_path = Path(course_folder) / f"{base_name}.txt"
    transcript_path.write_text(full_text)
    
    # Save combined JSON
    json_path = Path(course_folder) / f"{base_name}_timestamps.json"
    combined_json = {
        'text': full_text,
        'segments': all_segments,
        'duration': total_duration,
        'language': self.language or 'auto'
    }
    with open(json_path, 'w') as f:
        json.dump(combined_json, f, indent=2)
    
    print(f"\n\u2705 Combined transcript saved: {transcript_path}")
    print(f"\u2705 Combined timestamps saved: {json_path}")
    
    return {
        'transcript_path': str(transcript_path),
        'json_path': str(json_path),
        'text': full_text,
        'duration': total_duration,
        'segments': all_segments
    }

```

### How to Use

1.  **Replace the function:** Copy the entire method above and use it to replace the `_transcribe_chunked` method in `class_recorder/transcriber.py`.
2.  **Ensure `_split_audio` is also updated:** Make sure you have also updated the `_split_audio` method as described in the previous guide (`gemini_overlap_guide.md`).

With these two changes, your application is now fully equipped to handle overlapping chunks, which should give you a noticeable improvement in transcription quality for long recordings.
