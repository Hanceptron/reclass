"""Class Recorder - Lecture recording and processing tool"""

__version__ = "1.0.0"

from .recorder import AudioRecorder, list_devices
from .transcriber import WhisperTranscriber
from .summarizer import LLMSummarizer
from .config import config

__all__ = [
    'AudioRecorder',
    'WhisperTranscriber', 
    'LLMSummarizer',
    'config',
    'list_devices'
]
