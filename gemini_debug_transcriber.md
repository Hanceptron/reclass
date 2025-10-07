# Debugging the Transcriber

This guide provides a modified `_transcribe_chunked` function with added `print` statements to help diagnose why the output `.txt` file is empty.

### Instructions

1.  **Open the file:** `class_recorder/transcriber.py`
2.  **Replace the function:** Replace your entire `_transcribe_chunked` method with the code provided below.
3.  **Run the command:** Execute `recorder process recordings/stanford_ai/stanford_ai.mp3` again.
4.  **Copy the output:** Copy the entire output from your terminal and paste it back to me so I can analyze the results.

---

### `_transcribe_chunked` Method with Debugging

```python
# In class_recorder/transcriber.py

def _transcribe_chunked(self, audio_path, course_folder, base_name):
    """Split large files, transcribe overlapping chunks, and merge them."""
    print("\n[DEBUG] Starting _transcribe_chunked...")
    temp_dir = Path(config.get('storage.temp_dir', './temp'))
    temp_dir.mkdir(exist_ok=True)
    
    chunk_duration_min = 30
    overlap_min = 1
    overlap_sec = overlap_min * 60
    
    chunks = self._split_audio(audio_path, temp_dir, chunk_duration_min, overlap_min)
    
    print(f"\n[DEBUG] Split into {len(chunks)} overlapping chunks.")
    
    all_segments = []
    last_global_end_time = 0.0
    chunk_start_offset = 0.0

    for i, chunk in enumerate(chunks):
        print(f"\n[DEBUG] Processing chunk {i + 1}/{len(chunks)}: {chunk}")
        
        with open(chunk, 'rb') as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format=self.response_format,
                language=self.language,
                temperature=self.temperature
            )
        
        current_segments = self._extract_segments(response)
        print(f"[DEBUG] Chunk {i + 1} returned {len(current_segments)} segments.")

        if not current_segments:
            chunk.unlink()
            chunk_start_offset += (chunk_duration_min * 60) - overlap_sec
            continue

        segments_added_this_chunk = 0
        for segment in current_segments:
            segment_global_start = chunk_start_offset + segment['start']
            segment_global_end = chunk_start_offset + segment['end']

            if segment_global_start >= last_global_end_time:
                segment['start'] = segment_global_start
                segment['end'] = segment_global_end
                
                all_segments.append(segment)
                last_global_end_time = segment_global_end
                segments_added_this_chunk += 1
        
        print(f"[DEBUG] Added {segments_added_this_chunk} new segments from chunk {i + 1}.")
        
        chunk_start_offset += (chunk_duration_min * 60) - overlap_sec
        chunk.unlink()
    
    print(f"\n[DEBUG] Total segments collected: {len(all_segments)}")
    
    full_text = ' '.join([s['text'].strip() for s in all_segments])
    print(f"[DEBUG] Final text length: {len(full_text)}")
    # For safety, printing a small snippet of the final text
    print(f"[DEBUG] Final text snippet: {full_text[:200]}...")

    total_duration = all_segments[-1]['end'] if all_segments else 0.0
    
    transcript_path = Path(course_folder) / f"{base_name}.txt"
    print(f"[DEBUG] Writing final transcript to: {transcript_path}")
    transcript_path.write_text(full_text)
    
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