"""Audio recording with sounddevice - optimized for M3 Mac"""
import sounddevice as sd
import soundfile as sf
import signal
import sys
from pathlib import Path
from datetime import datetime
import queue
import shutil
import subprocess
import time

from .config import config

class GracefulKiller:
    """Handle Ctrl+C to save partial recordings"""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        print("\n⚠️  Stopping recording...")
        self.kill_now = True

class AudioRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.killer = GracefulKiller()
        self.q = queue.Queue()
        
        # Get settings from config
        self.sample_rate = config.get('recording.sample_rate', 16000)
        self.channels = config.get('recording.channels', 1)
        self.bitrate = config.get('recording.bitrate', '96k')
    
    def callback(self, indata, frames, time, status):
        """Audio callback - runs in separate thread"""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        self.q.put(indata.copy())
    
    def record(self, course_folder):
        """Record audio and return file paths"""
        # Generate initial filename with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        temp_wav = self.output_dir / f"temp_{date_str}.wav"
        
        print(f"\n🎙️  Recording started...")
        print(f"⚙️  Settings: {self.sample_rate}Hz, {self.channels} channel(s)")
        print(f"📁 Output folder: {Path(course_folder).resolve()}")
        print("Press Ctrl+C to stop\n")
        
        try:
            # Record to WAV (streaming)
            with sf.SoundFile(
                temp_wav,
                mode='w',
                samplerate=self.sample_rate,
                channels=self.channels,
                subtype='PCM_16'
            ) as file:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=2048,
                    callback=self.callback
                ):
                    print("⏺️  Recording... (Ctrl+C to stop)")
                    start_time = time.time()
                    last_update = start_time
                    while not self.killer.kill_now:
                        try:
                            chunk = self.q.get(timeout=0.25)
                        except queue.Empty:
                            chunk = None
                        if chunk is not None:
                            file.write(chunk)
                        now = time.time()
                        if now - last_update >= 1:
                            elapsed = int(now - start_time)
                            minutes, seconds = divmod(elapsed, 60)
                            print(f"\r⏱️  Elapsed: {minutes:02d}:{seconds:02d}", end='', flush=True)
                            last_update = now
                    # make sure the timer line does not swallow the next print
                    print()
            
            # Get recording info
            info = sf.info(temp_wav)
            duration = info.duration
            size_mb = temp_wav.stat().st_size / (1024 * 1024)
            
            print(f"\n✅ Recording complete!")
            print(f"📊 Duration: {int(duration//60)}m {int(duration%60)}s")
            print(f"💾 WAV size: {size_mb:.1f} MB")
            
            # Ask for class name
            class_name = input("\n📝 Enter class name (e.g., 'Derivatives'): ").strip()
            
            # Create final filename
            if class_name:
                base_name = f"{date_str}-{class_name.replace(' ', '-')}"
            else:
                base_name = date_str
            
            # Convert to M4A
            final_m4a = Path(course_folder) / f"{base_name}.m4a"
            final_m4a.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\n🔄 Converting to M4A...")
            try:
                self._convert_to_m4a(temp_wav, final_m4a)
            except Exception as convert_err:
                rescue_path = Path(course_folder) / f"{base_name}_raw.wav"
                try:
                    shutil.move(temp_wav, rescue_path)
                except Exception as rescue_err:
                    print(
                        f"⚠️ Failed to move raw WAV to safety: {rescue_err}",
                        file=sys.stderr
                    )
                else:
                    print(
                        f"⚠️ Conversion failed. Raw WAV preserved at {rescue_path}",
                        file=sys.stderr
                    )
                raise RuntimeError(
                    f"Conversion to M4A failed: {convert_err}"
                ) from convert_err
            else:
                # Clean up temp WAV only after successful conversion
                temp_wav.unlink()
            
            m4a_size = final_m4a.stat().st_size / (1024 * 1024)
            print(f"✅ Saved: {final_m4a}")
            print(f"💾 M4A size: {m4a_size:.1f} MB ({(size_mb/m4a_size):.1f}x smaller)")
            
            return {
                'audio_path': str(final_m4a),
                'base_name': base_name,
                'duration': duration,
                'course_folder': course_folder
            }
            
        except Exception as e:
            print(f"\n❌ Error: {e}", file=sys.stderr)
            if temp_wav.exists():
                temp_wav.unlink()
            raise
    
    def _convert_to_m4a(self, wav_path, m4a_path):
        """Convert WAV to M4A using ffmpeg"""
        cmd = [
            'ffmpeg', '-i', str(wav_path),
            '-c:a', 'aac',
            '-b:a', self.bitrate,
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-y',  # Overwrite
            str(m4a_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

def list_devices():
    """List available microphones"""
    print("\n🎤 Available microphones:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " [DEFAULT]" if i == sd.default.device[0] else ""
            print(f"[{i}] {device['name']}{default}")
            print(f"    Max channels: {device['max_input_channels']}")
            print(f"    Sample rate: {device['default_samplerate']} Hz")
    print("-" * 60)
