"""Whisper API transcription with automatic chunking"""
import json
import math
import subprocess
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .utils import build_frontmatter, extract_date_fragment

class WhisperTranscriber:
    # Whisper API limit: 25MB
    MAX_FILE_SIZE_MB = 24  # Stay slightly under limit
    
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.get('transcription.model', 'whisper-1')
        self.language = config.get('transcription.language')
        self.response_format = config.get('transcription.response_format', 'verbose_json')
        self.temperature = config.get('transcription.temperature', 0)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def transcribe(self, audio_path, course_folder, base_name):
        """Transcribe audio file, handling chunking if needed"""
        audio_path = Path(audio_path)
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        print(f"\n📝 Transcribing with Whisper API...")
        print(f"📊 File size: {file_size_mb:.1f} MB")
        
        # Check if chunking needed
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            print(f"⚠️  File exceeds {self.MAX_FILE_SIZE_MB}MB limit")
            print(f"🔪 Splitting into chunks...")
            return self._transcribe_chunked(audio_path, course_folder, base_name)
        else:
            return self._transcribe_single(audio_path, course_folder, base_name)
    
    def _transcribe_single(self, audio_path, course_folder, base_name):
        """Transcribe single file"""
        with open(audio_path, 'rb') as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format=self.response_format,
                language=self.language,
                temperature=self.temperature
            )
        
        segments = self._extract_segments(response)
        text = self._extract_text(response)
        if text is None:
            text = ""
        duration = self._extract_duration(response, self._probe_duration(audio_path))
        model_dump = self._serialize_response(response)

        # Save transcript
        transcript_path = Path(course_folder) / f"{base_name}.txt"
        with open(transcript_path, 'w') as f:
            f.write(text)

        # Save detailed JSON with timestamps
        json_path = Path(course_folder) / f"{base_name}_timestamps.json"
        with open(json_path, 'w') as f:
            json.dump(model_dump, f, indent=2)

        self._write_markdown_transcript(course_folder, base_name, text, duration)

        print(f"✅ Transcript saved: {transcript_path}")
        print(f"✅ Timestamps saved: {json_path}")

        return {
            'transcript_path': str(transcript_path),
            'json_path': str(json_path),
            'text': text,
            'duration': duration,
            'segments': segments
        }
    
    def _transcribe_chunked(self, audio_path, course_folder, base_name):
        """Split large files and transcribe chunks"""
        temp_dir = Path(config.get('storage.temp_dir', './temp'))
        temp_dir.mkdir(exist_ok=True)
        
        # Split file using ffmpeg
        chunks = self._split_audio(audio_path, temp_dir)

        print(f"📊 Split into {len(chunks)} chunks")

        # Transcribe each chunk
        all_text = []
        all_segments = []
        total_duration = 0.0
        
        for i, chunk_info in enumerate(chunks, 1):
            chunk_path = chunk_info['path']
            chunk_offset = chunk_info['offset']
            print(f"\n🔄 Transcribing chunk {i}/{len(chunks)}...")
            
            with open(chunk_path, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format=self.response_format,
                    language=self.language,
                    temperature=self.temperature
                )
            
            text = self._extract_text(response) or ""
            all_text.append(text)
            duration = self._extract_duration(response, 0.0)
            total_duration += duration
            
            for segment in self._extract_segments(response):
                segment['start'] += chunk_offset
                segment['end'] += chunk_offset
                all_segments.append(segment)
            
            # Clean up chunk
            chunk_path.unlink(missing_ok=True)
        
        if total_duration == 0:
            total_duration = self._probe_duration(audio_path)

        # Combine results
        full_text = '\n\n'.join(all_text)
        
        # Save combined transcript
        transcript_path = Path(course_folder) / f"{base_name}.txt"
        with open(transcript_path, 'w') as f:
            f.write(full_text)
        
        self._write_markdown_transcript(course_folder, base_name, full_text, total_duration)

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
        
        print(f"\n✅ Combined transcript saved: {transcript_path}")
        print(f"✅ Combined timestamps saved: {json_path}")
        
        return {
            'transcript_path': str(transcript_path),
            'json_path': str(json_path),
            'text': full_text,
            'duration': total_duration,
            'segments': all_segments
        }
    
    def _split_audio(self, audio_path, temp_dir):
        """Split audio into chunks that respect Whisper's file-size limits."""
        max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        audio_size = audio_path.stat().st_size
        total_duration = self._probe_duration(audio_path)

        target_chunks = max(1, math.ceil(audio_size / max_bytes))
        attempt = 0
        temp_dir.mkdir(parents=True, exist_ok=True)

        while attempt < 8:
            attempt += 1
            chunk_duration_sec = max(60, math.ceil(total_duration / target_chunks))

            # Clear previous attempt's chunks
            for existing in temp_dir.glob('chunk_*.m4a'):
                existing.unlink(missing_ok=True)

            chunks = []
            too_large = False
            index = 0
            while True:
                start_time = index * chunk_duration_sec
                if start_time >= total_duration:
                    break

                chunk_path = temp_dir / f"chunk_{index:03d}.m4a"
                split_cmd = [
                    'ffmpeg', '-i', str(audio_path),
                    '-ss', str(start_time),
                    '-t', str(chunk_duration_sec),
                    '-c', 'copy',
                    '-y',
                    str(chunk_path)
                ]
                subprocess.run(split_cmd, capture_output=True, check=True)

                if not chunk_path.exists() or chunk_path.stat().st_size == 0:
                    break

                chunk_size = chunk_path.stat().st_size
                if chunk_size > max_bytes:
                    too_large = True

                chunks.append({'path': chunk_path, 'offset': start_time})
                index += 1

            if chunks and not too_large:
                return chunks

            target_chunks += 1

        raise RuntimeError("Unable to split audio into chunks under the size limit. Consider lowering bitrate or shortening recording.")

    def _probe_duration(self, audio_path: Path) -> float:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def _write_markdown_transcript(self, course_folder, base_name, text, duration):
        course_name = Path(course_folder).name
        frontmatter = build_frontmatter(base_name, course_name, duration)
        date_fragment = extract_date_fragment(base_name)
        transcript_md_path = Path(course_folder) / f"Transcribe {date_fragment}.md"
        transcript_md_path.write_text(f"{frontmatter}\n\n{text}")

    def _extract_text(self, response):
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, 'text'):
            return getattr(response, 'text')
        if hasattr(response, 'get') and callable(getattr(response, 'get')):
            value = response.get('text')
            if value:
                return value
        return None

    def _extract_duration(self, response, fallback):
        if hasattr(response, 'duration') and getattr(response, 'duration') is not None:
            return float(getattr(response, 'duration'))
        if isinstance(response, dict) and response.get('duration') is not None:
            return float(response['duration'])
        return fallback
    
    def _serialize_response(self, response):
        if isinstance(response, str):
            return {'text': response}
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        if hasattr(response, 'to_dict'):
            return response.to_dict()
        return {
            'text': self._extract_text(response) or "",
            'segments': self._extract_segments(response),
            'duration': self._extract_duration(response, 0),
            'language': getattr(response, 'language', None) if hasattr(response, 'language') else None
        }
    
    def _extract_segments(self, response):
        raw_segments = getattr(response, 'segments', None)
        if not raw_segments:
            return []
        extracted = []
        for segment in raw_segments:
            if isinstance(segment, dict):
                extracted.append({
                    'id': segment.get('id'),
                    'start': float(segment.get('start', 0)),
                    'end': float(segment.get('end', 0)),
                    'text': segment.get('text', '')
                })
            else:
                extracted.append({
                    'id': getattr(segment, 'id', None),
                    'start': float(getattr(segment, 'start', 0)),
                    'end': float(getattr(segment, 'end', 0)),
                    'text': getattr(segment, 'text', '')
                })
        return extracted
