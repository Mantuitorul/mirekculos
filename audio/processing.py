#!/usr/bin/env python3
"""
Audio processing utilities for the pipeline.
Handles text-to-speech conversion, audio splitting, and clustering.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from pydub.silence import split_on_silence

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
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i}: {len(chunk)/1000:.2f}s")
    
    return chunks

def cluster_audio(chunks: List[AudioSegment], group_size: int = 3) -> List[AudioSegment]:
    """
    Clusters a list of audio chunks into groups of maximum `group_size` chunks.
    Each group is concatenated into a single AudioSegment.
    
    Args:
        chunks: List of audio chunks
        group_size: Maximum number of chunks in each cluster
        
    Returns:
        A list of concatenated audio clusters
    """
    logger.info(f"Clustering {len(chunks)} audio chunks with max group size {group_size}")
    
    if not chunks:
        logger.warning("No chunks to cluster, returning empty list")
        return []
        
    clusters = []
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i+group_size]
        if group:
            # Concatenate the group
            cluster = group[0]
            for chunk in group[1:]:
                cluster += chunk
                
            cluster_length = len(cluster) / 1000  # Convert to seconds
            clusters.append(cluster)
            
            logger.info(f"Created cluster {len(clusters)} with {len(group)} chunks, length: {cluster_length:.2f}s")
    
    logger.info(f"Created {len(clusters)} clusters in total")
    return clusters

def extract_audio_from_segments(segments: List[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract audio from video segments.
    
    Args:
        segments: List of segment information dictionaries
        output_dir: Directory to save extracted audio files
        
    Returns:
        Updated segments with audio paths
    """
    logger.info(f"Extracting audio from {len(segments)} video segments")
    
    # Create audio output directory
    audio_dir = output_dir / "extracted_audio"
    audio_dir.mkdir(exist_ok=True, parents=True)
    
    from moviepy.video.io.VideoFileClip import VideoFileClip
    
    # Extract audio from each segment
    for segment in segments:
        segment_path = segment.get("path")
        if not segment_path or not Path(segment_path).exists():
            logger.warning(f"Segment path not found: {segment_path}")
            continue
        
        segment_index = segment.get("order", 0)
        audio_path = str(audio_dir / f"audio_segment_{segment_index}.mp3")
        
        try:
            logger.info(f"Extracting audio from segment {segment_index} to {audio_path}")
            video_clip = VideoFileClip(segment_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path, codec="libmp3lame", verbose=False, logger=None)
            
            segment["audio_path"] = audio_path
            
            # Close clips
            audio_clip.close()
            video_clip.close()
            
            logger.info(f"Successfully extracted audio for segment {segment_index}")
            
        except Exception as e:
            logger.error(f"Error extracting audio from segment {segment_index}: {str(e)}")
    
    logger.info("Audio extraction complete")
    return segments