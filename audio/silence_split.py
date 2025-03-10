#!/usr/bin/env python3
# audio/silence_split.py
"""
Audio silence detection and splitting utilities.
Splits audio files into chunks based on silence detection.
"""

import logging
from typing import List
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Configure logging
logger = logging.getLogger(__name__)

def split_audio(
    audio_path: str, 
    silence_thresh: int = -50, 
    min_silence_len: int = 500, 
    keep_silence: int = 100
) -> List[AudioSegment]:
    """
    Splits an audio file into chunks based on silence.
    
    Args:
        audio_path: Path to the audio file
        silence_thresh: Silence threshold in dBFS
        min_silence_len: Minimum length of silence to be used for a split
        keep_silence: Amount of silence (in ms) to leave at the beginning and end of each chunk
        
    Returns:
        A list of audio chunks
    """
    logger.info(f"Splitting audio file: {audio_path}")
    logger.info(f"Parameters: silence_thresh={silence_thresh}, min_silence_len={min_silence_len}, keep_silence={keep_silence}")
    
    audio = AudioSegment.from_file(audio_path, format="mp3")
    logger.info(f"Audio length: {len(audio)/1000:.2f} seconds")
    
    chunks = split_on_silence(
        audio, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh, 
        keep_silence=keep_silence
    )
    
    # Handle the case where there are no clear silence breaks
    if not chunks and len(audio) > 0:
        logger.warning("No silence detected for splitting. Using the entire audio as a single chunk.")
        return [audio]
    
    # Log information about the chunks
    chunk_info = []
    for i, chunk in enumerate(chunks):
        chunk_info.append(f"Chunk {i}: {len(chunk)/1000:.2f}s")
    
    logger.info(f"Split into {len(chunks)} chunks: {', '.join(chunk_info)}")
    return chunks