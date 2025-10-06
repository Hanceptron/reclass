"""Whisper API transcription with automatic chunking"""
import json
import subprocess
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config

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
        text = getattr(response, 'text', "")
        duration = getattr(response, 'duration', 0)
        model_dump = self._serialize_response(response)
        
        # Save transcript
        transcript_path = Path(course_folder) / f"{base_name}.txt"
        with open(transcript_path, 'w') as f:
            f.write(text)
        
        # Save detailed JSON with timestamps
        json_path = Path(course_folder) / f"{base_name}_timestamps.json"
        with open(json_path, 'w') as f:
            json.dump(model_dump, f, indent=2)
        
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
        
        # Calculate chunk duration (aim for ~20MB chunks)
        chunk_duration_min = 30  # 30 min chunks (~21MB at 96kbps)
        
        # Split file using ffmpeg
        chunks = self._split_audio(audio_path, temp_dir, chunk_duration_min)
        
        print(f"📊 Split into {len(chunks)} chunks")
        
        # Transcribe each chunk
        all_text = []
        all_segments = []
        total_duration = 0.0
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n🔄 Transcribing chunk {i}/{len(chunks)}...")
            chunk_offset = (i - 1) * chunk_duration_min * 60
            
            with open(chunk, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format=self.response_format,
                    language=self.language,
                    temperature=self.temperature
                )
            
            text = getattr(response, 'text', "")
            all_text.append(text)
            duration = getattr(response, 'duration', 0)
            total_duration += duration
            
            for segment in self._extract_segments(response):
                segment['start'] += chunk_offset
                segment['end'] += chunk_offset
                all_segments.append(segment)
            
            # Clean up chunk
            chunk.unlink()
        
        # Combine results
        full_text = '\n\n'.join(all_text)
        
        # Save combined transcript
        transcript_path = Path(course_folder) / f"{base_name}.txt"
        with open(transcript_path, 'w') as f:
            f.write(full_text)
        
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
    
    def _split_audio(self, audio_path, temp_dir, chunk_duration_min):
        """Split audio into chunks using ffmpeg"""
        chunks = []
        chunk_duration_sec = chunk_duration_min * 60
        
        # Get total duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
        
        # Split into chunks
        num_chunks = int(total_duration / chunk_duration_sec) + 1
        
        for i in range(num_chunks):
            start_time = i * chunk_duration_sec
            chunk_path = temp_dir / f"chunk_{i:03d}.m4a"
            
            split_cmd = [
                'ffmpeg', '-i', str(audio_path),
                '-ss', str(start_time),
                '-t', str(chunk_duration_sec),
                '-c', 'copy',  # No re-encoding
                '-y',
                str(chunk_path)
            ]
            subprocess.run(split_cmd, capture_output=True, check=True)
            chunks.append(chunk_path)
        
        return chunks
    
    def _serialize_response(self, response):
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        if hasattr(response, 'to_dict'):
            return response.to_dict()
        return {
            'text': getattr(response, 'text', ""),
            'segments': self._extract_segments(response),
            'duration': getattr(response, 'duration', 0),
            'language': getattr(response, 'language', None)
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
