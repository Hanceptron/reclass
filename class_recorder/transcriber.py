"""Whisper API transcription with improved chunking and verification"""
import json
import math
import subprocess
from pathlib import Path
import hashlib

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .utils import build_frontmatter, clean_transcript_text, extract_date_fragment

class WhisperTranscriber:
    # Whisper API limit: 25MB, but leave buffer for safety
    MAX_FILE_SIZE_MB = 20  # More conservative limit
    
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.get('transcription.model', 'whisper-1')
        self.language = config.get('transcription.language')
        self.response_format = config.get('transcription.response_format', 'verbose_json')
        self.temperature = config.get('transcription.temperature', 0)
        self.sample_rate = config.get('recording.sample_rate', 16000)
        self.preprocess_audio = config.get('transcription.preprocess_audio', False)
        self.highpass_hz = config.get('transcription.highpass_hz', 100)
        self.loudnorm_enabled = config.get('transcription.loudnorm', True)
        self.loudnorm_settings = config.get(
            'transcription.loudnorm_settings',
            'I=-23:LRA=7:TP=-2'
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def transcribe(self, audio_path, course_folder, base_name):
        """Transcribe audio file with verification of complete processing."""
        audio_path = Path(audio_path)
        original_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        print(f"\n🎯 Transcribing with Whisper API...")
        print(f"📊 Original file size: {original_size_mb:.1f} MB")
        
        # Calculate file hash for verification
        file_hash = self._calculate_file_hash(audio_path)
        print(f"🔐 File hash: {file_hash[:8]}...")

        original_duration = self._probe_duration(audio_path)

        # Optional audio preprocessing for better ASR accuracy
        prepared_audio = self._prepare_audio_for_transcription(audio_path)
        working_path = prepared_audio or audio_path
        working_size_mb = working_path.stat().st_size / (1024 * 1024)
        actual_duration = self._probe_duration(working_path)

        if prepared_audio:
            print(f"📊 Working file size after preprocessing: {working_size_mb:.1f} MB")
        print(f"⏱️ Audio duration used for ASR: {actual_duration:.1f} seconds")

        # Transcribe based on size
        if working_size_mb > self.MAX_FILE_SIZE_MB:
            print(f"📦 File exceeds {self.MAX_FILE_SIZE_MB}MB - using smart chunking")
            result = self._transcribe_chunked_with_overlap(
                working_path, course_folder, base_name, actual_duration
            )
        else:
            result = self._transcribe_single(
                working_path, course_folder, base_name, actual_duration
            )
        
        # Verify completeness
        self._verify_transcription(result, actual_duration)
        
        # Clean up temp audio if created
        if prepared_audio and prepared_audio.exists():
            prepared_audio.unlink(missing_ok=True)
        
        return result
    
    def _transcribe_single(self, audio_path, course_folder, base_name, actual_duration):
        """Transcribe single file with progress tracking."""
        print("📤 Uploading to Whisper API...")
        
        with open(audio_path, 'rb') as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format=self.response_format,
                language=self.language,
                temperature=self.temperature,
                prompt="This is a classroom lecture. Include all spoken content."
            )
        
        segments = self._extract_segments(response)
        text = self._extract_text(response) or ""
        duration = self._extract_duration(response, actual_duration)
        
        # Calculate processing stats
        words_per_minute = len(text.split()) / (duration / 60) if duration > 0 else 0
        
        print(f"✅ Transcription complete!")
        print(f"📝 Words transcribed: {len(text.split())}")
        print(f"⚡ Words per minute: {words_per_minute:.0f}")
        
        # Save files
        transcript_path = Path(course_folder) / f"{base_name}.txt"
        cleaned_txt_path = Path(course_folder) / f"{base_name}_clean.txt"
        json_path = Path(course_folder) / f"{base_name}_timestamps.json"

        metadata = {
            'text': text,
            'segments': segments,
            'duration': duration,
            'word_count': len(text.split()),
            'language': self.language or 'auto-detected'
        }

        transcript_path.write_text(text)
        cleaned_text = clean_transcript_text(text)
        cleaned_txt_path.write_text(cleaned_text)

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self._write_markdown_transcripts(course_folder, base_name, text, cleaned_text, duration)

        print(f"💾 Saved transcript: {transcript_path}")
        print(f"💾 Saved clean transcript: {cleaned_txt_path}")
        print(f"💾 Saved timestamps: {json_path}")
        
        return {
            'transcript_path': str(transcript_path),
            'clean_transcript_path': str(cleaned_txt_path),
            'json_path': str(json_path),
            'text': text,
            'clean_text': cleaned_text,
            'duration': duration,
            'segments': segments,
            'word_count': len(text.split())
        }
    
    def _transcribe_chunked_with_overlap(self, audio_path, course_folder, base_name, actual_duration):
        """Improved chunking with overlap to not miss content."""
        temp_dir = Path(config.get('storage.temp_dir', './temp'))
        temp_dir.mkdir(exist_ok=True)
        
        # Create overlapping chunks
        chunks = self._create_overlapping_chunks(audio_path, temp_dir, actual_duration)
        print(f"📦 Created {len(chunks)} overlapping chunks")
        
        # Process chunks
        all_segments = []
        chunk_texts = []
        processed_duration = 0
        
        for i, chunk_info in enumerate(chunks, 1):
            chunk_path = chunk_info['path']
            chunk_start = chunk_info['start']
            chunk_end = chunk_info['end']
            overlap_start = chunk_info.get('overlap_start', 0)
            
            print(f"\n🔄 Processing chunk {i}/{len(chunks)} ({chunk_start:.0f}s - {chunk_end:.0f}s)")
            
            try:
                with open(chunk_path, 'rb') as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format=self.response_format,
                        language=self.language,
                        temperature=self.temperature
                    )
                
                chunk_text = self._extract_text(response) or ""
                chunk_segments = self._extract_segments(response)
                
                # Handle overlap - remove duplicate content
                if overlap_start > 0 and chunk_texts:
                    chunk_text = self._remove_overlap_duplicates(
                        chunk_texts[-1], chunk_text, overlap_start
                    )
                
                chunk_texts.append(chunk_text)
                
                # Adjust segment timestamps
                for segment in chunk_segments:
                    segment['start'] += chunk_start
                    segment['end'] += chunk_start
                    
                    # Only add segments that are after overlap point
                    if segment['start'] >= chunk_start + overlap_start:
                        all_segments.append(segment)
                
                processed_duration = chunk_end
                
            except Exception as e:
                print(f"  ⚠️ Error processing chunk {i}: {e}")
                # Don't fail completely, continue with other chunks
                continue
            
            finally:
                # Clean up chunk file
                chunk_path.unlink(missing_ok=True)
        
        # Combine results
        full_text = '\n'.join(chunk_texts)
        
        # Verify we processed most of the audio
        coverage = (processed_duration / actual_duration * 100) if actual_duration > 0 else 0
        print(f"\n📊 Coverage: {coverage:.1f}% of audio processed")
        
        if coverage < 95:
            print("⚠️ Warning: Less than 95% coverage achieved")
        
        # Save results
        transcript_path = Path(course_folder) / f"{base_name}.txt"
        cleaned_txt_path = Path(course_folder) / f"{base_name}_clean.txt"
        json_path = Path(course_folder) / f"{base_name}_timestamps.json"

        metadata = {
            'text': full_text,
            'segments': all_segments,
            'duration': actual_duration,
            'processed_duration': processed_duration,
            'coverage_percent': coverage,
            'word_count': len(full_text.split()),
            'chunks_processed': len(chunks)
        }

        transcript_path.write_text(full_text)
        cleaned_text = clean_transcript_text(full_text)
        cleaned_txt_path.write_text(cleaned_text)

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self._write_markdown_transcripts(course_folder, base_name, full_text, cleaned_text, actual_duration)

        print(f"✅ Combined transcript saved: {transcript_path}")
        print(f"✅ Clean transcript saved: {cleaned_txt_path}")
        print(f"📝 Total words: {len(full_text.split())}")
        
        return {
            'transcript_path': str(transcript_path),
            'clean_transcript_path': str(cleaned_txt_path),
            'json_path': str(json_path),
            'text': full_text,
            'clean_text': cleaned_text,
            'duration': actual_duration,
            'segments': all_segments,
            'word_count': len(full_text.split()),
            'coverage_percent': coverage
        }
    
    def _create_overlapping_chunks(self, audio_path, temp_dir, total_duration):
        """Create chunks with overlap to ensure no content is missed."""
        max_chunk_duration = 600  # 10 minutes max per chunk
        overlap_duration = 10  # 10 seconds overlap
        
        # Calculate optimal chunk size
        file_size = audio_path.stat().st_size
        estimated_chunks = math.ceil(file_size / (self.MAX_FILE_SIZE_MB * 1024 * 1024))
        chunk_duration = min(
            max_chunk_duration,
            total_duration / estimated_chunks
        )
        
        chunks = []
        current_time = 0
        index = 0
        
        while current_time < total_duration:
            chunk_start = max(0, current_time - overlap_duration if current_time > 0 else 0)
            chunk_end = min(current_time + chunk_duration, total_duration)
            overlap_start = overlap_duration if current_time > 0 else 0
            
            chunk_path = temp_dir / f"chunk_{index:03d}.m4a"
            
            # Extract chunk with ffmpeg
            cmd = [
                'ffmpeg', '-i', str(audio_path),
                '-ss', str(chunk_start),
                '-t', str(chunk_end - chunk_start),
                '-c:a', 'aac',
                '-b:a', '128k',
                '-y', str(chunk_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ⚠️ FFmpeg warning: {result.stderr[:100]}")
            
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunks.append({
                    'path': chunk_path,
                    'start': current_time,
                    'end': chunk_end,
                    'overlap_start': overlap_start
                })
            
            current_time = chunk_end
            index += 1
        
        return chunks
    
    def _remove_overlap_duplicates(self, previous_text, current_text, overlap_seconds):
        """Remove duplicate content from overlapping chunks."""
        # Simple approach: find common suffix/prefix
        # In practice, this is complex - using simple approach for now
        if len(current_text) < 100:
            return current_text
        
        # Look for repeating phrases at boundaries
        words_to_check = 20
        prev_words = previous_text.split()[-words_to_check:]
        curr_words = current_text.split()[:words_to_check * 2]
        
        # Find where overlap starts in current text
        for i in range(len(curr_words) - len(prev_words)):
            if curr_words[i:i+len(prev_words)] == prev_words:
                # Found overlap, remove it
                return ' '.join(curr_words[i+len(prev_words):])
        
        return current_text
    
    def _verify_transcription(self, result, expected_duration):
        """Verify that transcription is complete."""
        transcribed_duration = result.get('duration', 0)
        word_count = result.get('word_count', 0)
        
        # Expected words per minute for lecture (100-150 wpm typical)
        expected_min_words = (expected_duration / 60) * 80  # Conservative estimate
        
        print("\n🔍 Verification Report:")
        print(f"  Expected duration: {expected_duration:.1f}s")
        print(f"  Transcribed duration: {transcribed_duration:.1f}s")
        print(f"  Word count: {word_count}")
        print(f"  Expected minimum words: {expected_min_words:.0f}")
        
        if word_count < expected_min_words * 0.5:
            print("  ⚠️ WARNING: Transcript seems too short!")
            print("  Consider re-running with different settings")
        else:
            print("  ✅ Transcript length looks reasonable")
        
        if 'coverage_percent' in result:
            if result['coverage_percent'] < 95:
                print(f"  ⚠️ Coverage only {result['coverage_percent']:.1f}%")
            else:
                print(f"  ✅ Coverage: {result['coverage_percent']:.1f}%")
    
    def _calculate_file_hash(self, file_path):
        """Calculate file hash for verification."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _probe_duration(self, audio_path: Path) -> float:
        """Get accurate duration using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def _prepare_audio_for_transcription(self, audio_path: Path):
        """Optionally apply high-pass filtering and loudness normalization."""
        if not self.preprocess_audio:
            return None

        temp_dir = Path(config.get('storage.temp_dir', './temp'))
        temp_dir.mkdir(parents=True, exist_ok=True)

        output_path = temp_dir / f"preprocessed_{audio_path.stem}.wav"

        filters = []
        if self.highpass_hz:
            filters.append(f"highpass=f={self.highpass_hz}")
        if self.loudnorm_enabled:
            filters.append(f"loudnorm={self.loudnorm_settings}")

        cmd = ['ffmpeg', '-y', '-i', str(audio_path)]
        if filters:
            cmd += ['-af', ','.join(filters)]
        cmd += ['-ar', str(self.sample_rate), '-ac', '1', str(output_path)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = result.stderr.splitlines()[0] if result.stderr else 'unknown error'
            print(f"⚠️ Audio preprocessing failed: {message}")
            output_path.unlink(missing_ok=True)
            return None

        print(
            f"🔧 Audio preprocessing applied (high-pass {self.highpass_hz}Hz, "
            f"loudnorm {'enabled' if self.loudnorm_enabled else 'disabled'})"
        )
        return output_path

    def _write_markdown_transcripts(self, course_folder, base_name, raw_text, cleaned_text, duration):
        """Write both raw and cleaned markdown transcripts."""
        course_name = Path(course_folder).name
        frontmatter = build_frontmatter(base_name, course_name, duration)
        date_fragment = extract_date_fragment(base_name)

        def _compose_markdown(content_text: str) -> str:
            word_count = len(content_text.split())
            reading_time = max(1, round(word_count / 200)) if word_count else 0
            header = f"{frontmatter}\n\n"
            header += f"*Words: {word_count} | Reading time: {reading_time} min*\n\n"
            header += "---\n\n"
            return header + content_text

        raw_md_path = Path(course_folder) / f"Transcript-{date_fragment}.md"
        clean_md_path = Path(course_folder) / f"Transcript-{date_fragment}-clean.md"

        raw_md_path.write_text(_compose_markdown(raw_text))
        clean_md_path.write_text(_compose_markdown(cleaned_text))

        return raw_md_path, clean_md_path
    
    def _extract_text(self, response):
        """Extract text from response object."""
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, 'text'):
            return getattr(response, 'text', '')
        return ''
    
    def _extract_duration(self, response, fallback):
        """Extract duration from response."""
        if hasattr(response, 'duration'):
            return float(getattr(response, 'duration', fallback))
        return fallback
    
    def _extract_segments(self, response):
        """Extract segments with timestamps."""
        segments = getattr(response, 'segments', [])
        extracted = []
        for segment in segments:
            if isinstance(segment, dict):
                extracted.append({
                    'start': float(segment.get('start', 0)),
                    'end': float(segment.get('end', 0)),
                    'text': segment.get('text', '')
                })
            else:
                extracted.append({
                    'start': float(getattr(segment, 'start', 0)),
                    'end': float(getattr(segment, 'end', 0)),
                    'text': getattr(segment, 'text', '')
                })
        return extracted
