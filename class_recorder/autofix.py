#!/usr/bin/env python3
"""Auto-detect and fix common issues with Class Recorder"""

import os
import sys
import subprocess
from pathlib import Path
import json
import shutil

class AutoFixer:
    def __init__(self):
        self.issues = []
        self.fixes_applied = []
    
    def run(self):
        """Run all checks and fixes."""
        print("🔧 Class Recorder Auto-Fix Tool\n")
        print("=" * 50)
        
        self.check_dependencies()
        self.check_environment()
        self.check_config()
        self.check_audio_system()
        self.check_disk_space()
        self.cleanup_temp_files()
        
        self.report()
    
    def check_dependencies(self):
        """Check if all required tools are installed."""
        print("\n📦 Checking dependencies...")
        
        # Check ffmpeg
        if not shutil.which('ffmpeg'):
            self.issues.append("FFmpeg not installed")
            print("  ❌ FFmpeg missing")
            if sys.platform == 'darwin':  # macOS
                print("  💡 To fix: brew install ffmpeg")
        else:
            print("  ✅ FFmpeg installed")
        
        # Check Python packages
        required_packages = [
            'click', 'sounddevice', 'soundfile', 
            'openai', 'tenacity', 'pyyaml', 'python-dotenv'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        
        if missing:
            self.issues.append(f"Missing Python packages: {', '.join(missing)}")
            print(f"  ❌ Missing packages: {', '.join(missing)}")
            print(f"  💡 To fix: pip install {' '.join(missing)}")
        else:
            print("  ✅ All Python packages installed")
    
    def check_environment(self):
        """Check environment variables."""
        print("\n🔐 Checking environment...")
        
        env_file = Path('.env')
        if not env_file.exists():
            self.issues.append(".env file not found")
            print("  ❌ .env file missing")
            
            # Offer to create template
            if input("  Create .env template? (y/n): ").lower() == 'y':
                env_content = """# API Keys for Class Recorder
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
"""
                env_file.write_text(env_content)
                self.fixes_applied.append("Created .env template")
                print("  ✅ Created .env template - please add your API keys")
        else:
            # Check if keys are set
            from dotenv import load_dotenv
            load_dotenv()
            
            if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here':
                self.issues.append("OPENAI_API_KEY not configured")
                print("  ⚠️ OPENAI_API_KEY not set properly")
            else:
                print("  ✅ OpenAI API key configured")
            
            if not os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENROUTER_API_KEY') == 'your_openrouter_api_key_here':
                print("  ⚠️ OPENROUTER_API_KEY not set (optional)")
    
    def check_config(self):
        """Check and optimize config.yaml."""
        print("\n⚙️ Checking configuration...")
        
        config_file = Path('config.yaml')
        if not config_file.exists():
            self.issues.append("config.yaml not found")
            print("  ❌ config.yaml missing")
            
            if input("  Create optimized config? (y/n): ").lower() == 'y':
                self.create_optimized_config()
                self.fixes_applied.append("Created optimized config.yaml")
                print("  ✅ Created optimized config.yaml")
        else:
            # Check if using optimal settings
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            sample_rate = config.get('recording', {}).get('sample_rate', 16000)
            if sample_rate < 44100:
                print(f"  ⚠️ Low sample rate ({sample_rate} Hz)")
                if input("  Upgrade to 48000 Hz? (y/n): ").lower() == 'y':
                    config['recording']['sample_rate'] = 48000
                    with open(config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    self.fixes_applied.append("Upgraded audio quality to 48000 Hz")
                    print("  ✅ Upgraded to 48000 Hz")
            else:
                print(f"  ✅ Good sample rate ({sample_rate} Hz)")
            
            # Check temperature for less hallucination
            temp = config.get('summarization', {}).get('temperature', 0.3)
            if temp > 0.2:
                print(f"  ⚠️ High temperature ({temp}) may cause hallucinations")
                if input("  Lower to 0.1? (y/n): ").lower() == 'y':
                    config['summarization']['temperature'] = 0.1
                    with open(config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    self.fixes_applied.append("Lowered temperature for accuracy")
    
    def check_audio_system(self):
        """Check audio system configuration."""
        print("\n🎙️ Checking audio system...")
        
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            # Find default input device
            default_input = sd.default.device[0]
            if default_input is not None:
                device = devices[default_input]
                print(f"  ✅ Default microphone: {device['name']}")
                print(f"     Max channels: {device['max_input_channels']}")
                print(f"     Default sample rate: {device['default_samplerate']} Hz")
            else:
                self.issues.append("No default microphone set")
                print("  ❌ No default microphone")
                print("  💡 Check System Settings > Sound > Input")
            
        except Exception as e:
            self.issues.append(f"Audio system error: {e}")
            print(f"  ❌ Could not check audio: {e}")
    
    def check_disk_space(self):
        """Check available disk space."""
        print("\n💾 Checking disk space...")
        
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        
        if free_gb < 5:
            self.issues.append(f"Low disk space: {free_gb:.1f} GB")
            print(f"  ⚠️ Low disk space: {free_gb:.1f} GB free")
            print("  💡 Each hour of recording needs ~100 MB")
        else:
            print(f"  ✅ Disk space: {free_gb:.1f} GB free")
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        print("\n🧹 Checking for temp files...")
        
        temp_dir = Path('./temp')
        if temp_dir.exists():
            temp_files = list(temp_dir.glob('*.wav')) + list(temp_dir.glob('chunk_*.m4a'))
            
            if temp_files:
                total_size = sum(f.stat().st_size for f in temp_files) / (1024**2)
                print(f"  Found {len(temp_files)} temp files ({total_size:.1f} MB)")
                
                if input("  Clean up temp files? (y/n): ").lower() == 'y':
                    for f in temp_files:
                        f.unlink()
                    self.fixes_applied.append(f"Cleaned {len(temp_files)} temp files")
                    print("  ✅ Temp files cleaned")
            else:
                print("  ✅ No temp files found")
    
    def create_optimized_config(self):
        """Create an optimized config.yaml file."""
        config = {
            'recording': {
                'sample_rate': 48000,
                'channels': 1,
                'bitrate': '128k',
                'blocksize': 2048
            },
            'transcription': {
                'model': 'whisper-1',
                'language': 'en',
                'response_format': 'verbose_json',
                'temperature': 0
            },
            'summarization': {
                'model': 'google/gemini-2.0-flash-exp',
                'max_tokens': 4000,
                'temperature': 0.1,
                'chunk_chars': 8000,
                'chunk_overlap_paragraphs': 2
            },
            'storage': {
                'temp_dir': './temp',
                'vault_dir': '~/Documents/Obsidian/ClassNotes'
            }
        }
        
        import yaml
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def report(self):
        """Print final report."""
        print("\n" + "=" * 50)
        print("📋 REPORT\n")
        
        if not self.issues and not self.fixes_applied:
            print("✅ Everything looks good! Ready to record.")
        else:
            if self.fixes_applied:
                print("✅ Fixes Applied:")
                for fix in self.fixes_applied:
                    print(f"  • {fix}")
                print()
            
            if self.issues:
                print("⚠️ Issues Remaining:")
                for issue in self.issues:
                    print(f"  • {issue}")
        
        print("\n" + "=" * 50)
        print("\nNext steps:")
        if self.issues:
            print("1. Fix remaining issues above")
            print("2. Run 'python autofix.py' again to verify")
            print("3. Test with: python -m class_recorder record ./test")
        else:
            print("1. Test recording: python test_quality.py audio")
            print("2. Start recording: python -m class_recorder record ./Classes/YourClass")
            print("3. After recording: python test_quality.py all ./Classes/YourClass")

def main():
    """Run the auto-fixer."""
    fixer = AutoFixer()
    fixer.run()

if __name__ == '__main__':
    main()