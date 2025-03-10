#!/usr/bin/env python3
# audio/generation.py
"""
Audio generation and processing utilities for the pipeline.
Handles text-to-speech conversion, audio splitting, and clustering.
"""

import logging
from pathlib import Path
from typing import List, Union
from elevenlabs.client import ElevenLabs

from audio.silence_split import split_audio
from audio.clustering import cluster_audio

# Configure logging
logger = logging.getLogger(__name__)

# Default ElevenLabs voice ID
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

async def generate_audio(
    text: str,
    output_dir: Path,
    elevenlabs_api_key: str,
    cluster_size: int = 3,
    silence_thresh: int = -50,
    min_silence_len: int = 500,
    keep_silence: int = 100
) -> List[str]:
    """
    Generates audio from text using ElevenLabs, splits by silence, clusters, and saves to files.
    
    Args:
        text: The text to convert to speech
        output_dir: Directory to save audio files
        elevenlabs_api_key: ElevenLabs API key
        cluster_size: Number of audio segments per cluster
        silence_thresh: Silence threshold for splitting
        min_silence_len: Minimum silence length for splitting
        keep_silence: Amount of silence to keep
        
    Returns:
        Paths to the generated audio files
    """
    try:
        temp_filename = output_dir / "temp_audio.mp3"
        
        # Initialize ElevenLabs client
        elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
        
        logger.info("Generating audio with ElevenLabs...")
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=DEFAULT_VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        # Handle both iterator and direct response cases
        audio_data = b''.join(audio_generator) if hasattr(audio_generator, '__iter__') else audio_generator
        
        # Save the initial audio file
        with open(temp_filename, "wb") as f:
            f.write(audio_data)
            
        logger.info(f"Initial audio saved to {temp_filename}")
        
        # Split the audio into chunks based on silence
        chunks = split_audio(
            str(temp_filename), 
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            keep_silence=keep_silence
        )
        logger.info(f"Split audio into {len(chunks)} chunks")
        
        # Then cluster those chunks
        clusters = cluster_audio(chunks, group_size=cluster_size)
        logger.info(f"Grouped chunks into {len(clusters)} clusters")
        
        # Save each cluster
        output_files = []
        for i, cluster in enumerate(clusters):
            cluster_filename = output_dir / f"audio_cluster_{i}.mp3"
            cluster.export(str(cluster_filename), format="mp3")
            output_files.append(str(cluster_filename))
            logger.info(f"Saved audio cluster {i} to {cluster_filename}")
        
        # Clean up temporary file
        temp_filename.unlink(missing_ok=True)
        
        logger.info(f"Audio generated and clustered successfully")
        return output_files
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise