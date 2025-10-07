#!/usr/bin/env python3
"""Test script to verify audio quality and processing completeness"""

import sys
from pathlib import Path
import sounddevice as sd
import numpy as np
import time
import json

def test_audio_quality():
    """Test microphone quality settings."""
    print("🎙️ Testing Audio Quality Settings\n")
    print("=" * 50)
    
    # Test different sample rates
    sample_rates = [16000, 44100, 48000]
    
    for rate in sample_rates:
        print(f"\nTesting {rate} Hz:")
        try:
            # Record 2 seconds
            duration = 2
            recording = sd.rec(
                int(duration * rate),
                samplerate=rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Analyze
            volume = np.abs(recording).mean()
            peak = np.abs(recording).max()
            
            print(f"  ✅ Recording successful")
            print(f"  📊 Average volume: {volume:.4f}")
            print(f"  📈 Peak volume: {peak:.4f}")
            
            if volume < 0.001:
                print(f"  ⚠️ WARNING: Very low volume detected!")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("Recommended: Use 48000 Hz for best quality")

def test_transcript_completeness(transcript_file):
    """Verify transcript completeness."""
    print(f"\n📄 Analyzing Transcript: {transcript_file}\n")
    print("=" * 50)
    
    transcript_path = Path(transcript_file)
    
    if not transcript_path.exists():
        print("❌ File not found!")
        return
    
    # Read transcript
    with open(transcript_path, 'r') as f:
        text = f.read()
    
    # Check for JSON metadata
    json_path = transcript_path.parent / f"{transcript_path.stem}_timestamps.json"
    
    # Basic stats
    word_count = len(text.split())
    line_count = len(text.splitlines())
    char_count = len(text)
    
    print(f"📊 Basic Statistics:")
    print(f"  Words: {word_count:,}")
    print(f"  Lines: {line_count:,}")
    print(f"  Characters: {char_count:,}")
    
    # Check for common issues
    print(f"\n🔍 Quality Checks:")
    
    # Check for [UNCLEAR] or [inaudible] markers
    unclear_count = text.lower().count('[unclear]') + text.lower().count('[inaudible]')
    if unclear_count > 0:
        print(f"  ⚠️ Found {unclear_count} unclear sections")
    else:
        print(f"  ✅ No unclear sections marked")
    
    # Check for repetition (might indicate chunking issues)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    duplicates = len(sentences) - len(set(sentences))
    if duplicates > 0:
        print(f"  ⚠️ Found {duplicates} duplicate sentences")
    else:
        print(f"  ✅ No duplicate sentences")
    
    # Check JSON metadata if exists
    if json_path.exists():
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📋 Metadata:")
        print(f"  Duration: {metadata.get('duration', 0):.1f} seconds")
        print(f"  Segments: {len(metadata.get('segments', []))}")
        
        if 'coverage_percent' in metadata:
            coverage = metadata['coverage_percent']
            if coverage < 95:
                print(f"  ⚠️ Coverage: {coverage:.1f}% (should be >95%)")
            else:
                print(f"  ✅ Coverage: {coverage:.1f}%")
        
        if 'chunks_processed' in metadata:
            print(f"  Chunks processed: {metadata['chunks_processed']}")
    
    # Words per minute estimate (assuming average speaking rate)
    if json_path.exists() and 'duration' in metadata:
        duration_min = metadata['duration'] / 60
        wpm = word_count / duration_min if duration_min > 0 else 0
        print(f"\n⚡ Words per minute: {wpm:.0f}")
        if wpm < 80:
            print(f"  ⚠️ Low WPM - might be missing content")
        elif wpm > 200:
            print(f"  ⚠️ High WPM - might have errors")
        else:
            print(f"  ✅ WPM looks reasonable for lecture")

def test_summary_quality(summary_file):
    """Check summary for potential issues."""
    print(f"\n📝 Analyzing Summary: {summary_file}\n")
    print("=" * 50)
    
    summary_path = Path(summary_file)
    if not summary_path.exists():
        print("❌ File not found!")
        return
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    print("🔍 Checking for common issues:")
    
    # Check for placeholder text
    issues = {
        '[NUMBER]': 'Unverified numbers',
        '[UNCLEAR]': 'Unclear sections',
        '[unclear]': 'Unclear sections',
        'NOT MENTIONED': 'Missing information markers',
        'Coverage:': 'Has coverage statistics'
    }
    
    for marker, description in issues.items():
        count = content.count(marker)
        if count > 0:
            if marker == 'Coverage:':
                print(f"  ✅ {description}")
            else:
                print(f"  ⚠️ Found {count} instances of {description}")
    
    # Check structure
    headers = [line for line in content.splitlines() if line.startswith('#')]
    print(f"\n📑 Structure:")
    print(f"  Headers found: {len(headers)}")
    if len(headers) < 3:
        print(f"  ⚠️ Very few headers - might be too condensed")
    
    # Word count
    word_count = len(content.split())
    print(f"  Word count: {word_count:,}")

def main():
    """Main test function."""
    print("\n🔧 Class Recorder Quality Test Tool\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_quality.py audio     - Test audio settings")
        print("  python test_quality.py transcript <file>  - Verify transcript")
        print("  python test_quality.py summary <file>     - Check summary")
        print("  python test_quality.py all <folder>       - Test all in folder")
        return
    
    command = sys.argv[1]
    
    if command == 'audio':
        test_audio_quality()
    
    elif command == 'transcript' and len(sys.argv) > 2:
        test_transcript_completeness(sys.argv[2])
    
    elif command == 'summary' and len(sys.argv) > 2:
        test_summary_quality(sys.argv[2])
    
    elif command == 'all' and len(sys.argv) > 2:
        folder = Path(sys.argv[2])
        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            return
        
        # Find latest files
        transcripts = list(folder.glob("*.txt"))
        summaries = list(folder.glob("*-notes.md"))
        
        if transcripts:
            latest_transcript = max(transcripts, key=lambda p: p.stat().st_mtime)
            test_transcript_completeness(latest_transcript)
        
        if summaries:
            latest_summary = max(summaries, key=lambda p: p.stat().st_mtime)
            test_summary_quality(latest_summary)
    
    else:
        print("❌ Invalid command. Run without arguments to see usage.")

if __name__ == '__main__':
    main()