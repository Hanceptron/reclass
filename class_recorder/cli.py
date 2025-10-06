"""CLI interface using Click"""
import sys
from pathlib import Path

import click

from .config import config
from .recorder import AudioRecorder, list_devices
from .summarizer import LLMSummarizer
from .transcriber import WhisperTranscriber

@click.group()
def cli():
    """🎓 Class Recorder - Record, transcribe, and summarize lectures"""
    pass

@cli.command()
@click.argument('course_folder', type=click.Path())
@click.option('--device', '-d', type=int, help='Audio device ID')
def record(course_folder, device):
    """Record a lecture and automatically transcribe + summarize"""
    
    if device is not None:
        import sounddevice as sd
        sd.default.device = device
    
    try:
        # Record
        recorder = AudioRecorder(config.get('storage.temp_dir', './temp'))
        result = recorder.record(course_folder)
        
        # Ask if user wants to process now
        if click.confirm('\n🔄 Transcribe and summarize now?', default=True):
            # Transcribe
            transcriber = WhisperTranscriber()
            trans_result = transcriber.transcribe(
                result['audio_path'],
                result['course_folder'],
                result['base_name']
            )
            
            # Summarize
            summarizer = LLMSummarizer()
            summarizer.summarize(
                trans_result['text'],
                result['course_folder'],
                result['base_name'],
                metadata={
                    'course': Path(course_folder).name,
                    'duration': result['duration']
                }
            )
            
            click.secho("\n✨ All done! Check your Obsidian vault.", fg='green')
        else:
            click.echo(f"\n💾 Audio saved. Process later with:")
            click.echo(f"   python -m class_recorder process {result['audio_path']}")
    
    except KeyboardInterrupt:
        click.secho("\n⚠️  Cancelled by user", fg='yellow')
        sys.exit(1)
    except Exception as e:
        click.secho(f"\n❌ Error: {e}", fg='red')
        sys.exit(1)

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
def process(audio_file):
    """Process an existing audio file (transcribe + summarize)"""
    
    audio_path = Path(audio_file)
    course_folder = audio_path.parent
    base_name = audio_path.stem
    
    try:
        # Transcribe
        transcriber = WhisperTranscriber()
        trans_result = transcriber.transcribe(
            str(audio_path),
            str(course_folder),
            base_name
        )
        
        # Summarize
        summarizer = LLMSummarizer()
        summarizer.summarize(
            trans_result['text'],
            str(course_folder),
            base_name,
            metadata={
                'course': course_folder.name,
                'duration': trans_result.get('duration', 0)
            }
        )
        
        click.secho("\n✨ Processing complete!", fg='green')
        
    except Exception as e:
        click.secho(f"\n❌ Error: {e}", fg='red')
        sys.exit(1)

@cli.command()
def devices():
    """List available microphones"""
    list_devices()

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
def transcribe_only(audio_file):
    """Transcribe audio without summarizing"""
    
    audio_path = Path(audio_file)
    course_folder = audio_path.parent
    base_name = audio_path.stem
    
    try:
        transcriber = WhisperTranscriber()
        transcriber.transcribe(
            str(audio_path),
            str(course_folder),
            base_name
        )
        click.secho("\n✅ Transcription complete!", fg='green')
        
    except Exception as e:
        click.secho(f"\n❌ Error: {e}", fg='red')
        sys.exit(1)

@cli.command()
@click.argument('transcript_file', type=click.Path(exists=True))
def summarize_only(transcript_file):
    """Summarize an existing transcript"""
    
    transcript_path = Path(transcript_file)
    course_folder = transcript_path.parent
    base_name = transcript_path.stem
    
    try:
        with open(transcript_path) as f:
            transcript_text = f.read()
        
        summarizer = LLMSummarizer()
        summarizer.summarize(
            transcript_text,
            str(course_folder),
            base_name,
            metadata={'course': course_folder.name}
        )
        
        click.secho("\n✅ Summary complete!", fg='green')
        
    except Exception as e:
        click.secho(f"\n❌ Error: {e}", fg='red')
        sys.exit(1)

if __name__ == '__main__':
    cli()
