#!/usr/bin/env python3
# pipeline_runner.py
"""
Main orchestrator for the text-to-video generation pipeline.
Coordinates between the audio, video, text, and utility modules.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Import modules
from utils.config import load_environment, setup_logging, ensure_output_dir
from audio.generation import generate_audio
from audio.upload import upload_audio_to_public_url
from text.segmentation import split_text_into_chunks
from text.clustering import cluster_text_chunks
from video.heygen_client import create_heygen_video, poll_video_status, download_video
from video.merger import merge_videos

# Setup logging
logger = logging.getLogger(__name__)
setup_logging()

# Default output directory
OUTPUT_DIR = Path("output")

class VideoConfig(BaseModel):
    """Configuration for video generation pipeline"""
    text: str = Field(..., description="Text to convert to speech and video")
    front_avatar_id: str = Field(..., description="Avatar ID for front pose")
    side_avatar_id: str = Field(..., description="Avatar ID for side pose")
    avatar_style: str = Field("normal", description="Avatar style (normal, happy, etc.)")
    background_color: str = Field("#008000", description="Background color in hex format")
    width: int = Field(1280, description="Video width in pixels")
    height: int = Field(720, description="Video height in pixels")
    output_filename: str = Field("final_output.mp4", description="Final output video filename")
    cluster_size: int = Field(3, description="Number of audio/text chunks per cluster")
    
    # Audio mode settings (ElevenLabs)
    silence_threshold: int = Field(-50, description="Silence threshold in dBFS for audio splitting")
    min_silence_len: int = Field(500, description="Minimum silence length in ms for audio splitting")
    keep_silence: int = Field(100, description="Amount of silence to keep in ms")
    
    # Text mode settings (HeyGen Voice)
    use_heygen_voice: bool = Field(False, description="Whether to use HeyGen's voice API instead of ElevenLabs")
    heygen_voice_id: Optional[str] = Field(None, description="Voice ID for HeyGen's TTS")
    heygen_emotion: Optional[str] = Field(None, description="Voice emotion for HeyGen's TTS (Excited, Friendly, Serious, Soothing, Broadcaster)")

    # B-roll post-processing options
    use_broll: bool = Field(False, description="Whether to add B-roll footage to the final video")
    broll_count: int = Field(3, description="Number of B-roll segments to insert")
    broll_duration: float = Field(5.0, description="Duration of each B-roll segment in seconds")
    broll_text_english: Optional[str] = Field(None, description="Text in English for B-roll keyword extraction")
    broll_orientation: str = Field("landscape", description="B-roll video orientation (landscape, portrait, square)")
    broll_video_size: str = Field("medium", description="B-roll video size (large=4K, medium=Full HD, small=HD)")

async def poll_and_download_video(video_id: str, api_key: str, output_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Poll video status until complete, then download it.
    
    Args:
        video_id: The ID of the video to poll
        api_key: HeyGen API key
        output_path: Path to save the video
        metadata: Dictionary containing metadata about the video segment
        
    Returns:
        Dict containing the video path and ordering information
    """
    status = await poll_video_status(video_id, api_key)
    video_url = status.get("video_url")
    if not video_url:
        raise ValueError(f"No video URL in status for video {video_id}")
    
    path = await download_video(video_url, str(output_path))
    
    # Return path and original ordering information
    return {
        "path": path,
        "order": metadata["original_order"]
    }

async def run_pipeline(
    text: str,
    front_avatar_id: str,
    side_avatar_id: str,
    avatar_style: str = "normal",
    background_color: str = "#008000",
    width: int = 1280,
    height: int = 720,
    output_filename: str = "final_output.mp4",
    output_dir: Optional[Path] = None,
    cluster_size: int = 3,
    silence_threshold: int = -50,
    min_silence_len: int = 500,
    keep_silence: int = 100,
    use_heygen_voice: bool = False,
    heygen_voice_id: Optional[str] = None,
    heygen_emotion: Optional[str] = None,
    use_broll: bool = False,
    broll_count: int = 3,
    broll_duration: float = 5.0,
    broll_text_english: Optional[str] = None,
    broll_orientation: str = "landscape",
    broll_video_size: str = "medium"
) -> Dict[str, Any]:
    """
    Run the entire text-to-video pipeline with optional B-roll post-processing.
    
    Args:
        text: The text to convert to speech and video
        front_avatar_id: Avatar ID for front pose
        side_avatar_id: Avatar ID for side pose
        avatar_style: Avatar style (normal, happy, etc.)
        background_color: Background color in hex format
        width: Video width in pixels
        height: Video height in pixels
        output_filename: Final output video filename
        output_dir: Directory to save output files
        cluster_size: Number of audio/text chunks per cluster
        silence_threshold: Silence threshold in dBFS for audio splitting
        min_silence_len: Minimum silence length in ms for audio splitting
        keep_silence: Amount of silence to keep in ms
        use_heygen_voice: Whether to use HeyGen's voice API instead of ElevenLabs
        heygen_voice_id: Voice ID for HeyGen's TTS (required if use_heygen_voice is True)
        use_broll: Whether to add B-roll footage to the final video
        broll_count: Number of B-roll segments to insert
        broll_duration: Duration of each B-roll segment in seconds
        broll_text_english: Text in English for B-roll keyword extraction (required if use_broll is True and text is not in English)
        broll_orientation: B-roll video orientation (landscape, portrait, square)
        broll_video_size: B-roll video size (large=4K, medium=Full HD, small=HD)
        
    Returns:
        Dict with success status, final video path, and other details
    """
    # Setup config object
    config = VideoConfig(
        text=text,
        front_avatar_id=front_avatar_id,
        side_avatar_id=side_avatar_id,
        avatar_style=avatar_style,
        background_color=background_color,
        width=width,
        height=height,
        output_filename=output_filename,
        cluster_size=cluster_size,
        silence_threshold=silence_threshold,
        min_silence_len=min_silence_len,
        keep_silence=keep_silence,
        use_heygen_voice=use_heygen_voice,
        heygen_voice_id=heygen_voice_id,
        heygen_emotion=heygen_emotion,
        use_broll=use_broll,
        broll_count=broll_count,
        broll_duration=broll_duration,
        broll_text_english=broll_text_english,
        broll_orientation=broll_orientation,
        broll_video_size=broll_video_size
    )

    if config.use_broll and not config.broll_text_english and not _is_text_english(config.text):
        logger.warning("Text appears to be non-English. For best B-roll results, provide English text via broll_text_english parameter.")
    
    
    # Validate config
    if config.use_heygen_voice and not config.heygen_voice_id:
        raise ValueError("heygen_voice_id is required when use_heygen_voice is True")
    
    # Setup output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR
    ensure_output_dir(output_dir)
    
    # Load API keys
    api_keys = load_environment()
    
    try:
        total_clusters = 0
        video_ids = []
        video_paths = []
        
        if config.use_heygen_voice:
            # Text-based approach using HeyGen's voice API
            logger.info("Using HeyGen voice API for text-to-speech")
            
            # Split text into chunks and clusters
            logger.info("Segmenting and clustering text...")
            text_chunks = split_text_into_chunks(config.text)
            text_clusters = cluster_text_chunks(text_chunks, config.cluster_size)
            total_clusters = len(text_clusters)
            
            logger.info(f"Created {total_clusters} text clusters")
            
            # Process each cluster fully and sequentially
            video_downloads = []
            
            # Process clusters in their original order
            for cluster in text_clusters:
                # Determine avatar pose (based on reversed position - last chunk gets FRONT_MODEL_ID)
                reversed_index = total_clusters - 1 - cluster["original_order"]
                current_avatar_id = (
                    config.front_avatar_id if reversed_index % 2 == 0 
                    else config.side_avatar_id
                )
                
                logger.info(f"Processing text cluster {cluster['original_order']+1}/{total_clusters}: "
                           f"Using avatar {current_avatar_id}")
                
                # Store metadata for later ordering
                metadata = {
                    "original_order": cluster["original_order"],
                    "reversed_index": reversed_index,
                    "avatar_id": current_avatar_id
                }
                
                # Step 1: Create video
                logger.info(f"Creating video for text cluster {cluster['original_order']+1}/{total_clusters}...")
                try:
                    video_id = await create_heygen_video(
                        avatar_id=current_avatar_id,
                        avatar_style=config.avatar_style,
                        background_color=config.background_color,
                        width=config.width,
                        height=config.height,
                        api_key=api_keys["heygen_api_key"],
                        input_text=cluster["text"],
                        voice_id=config.heygen_voice_id,
                        emotion=config.heygen_emotion
                    )
                    
                    # Step 2: Poll video status
                    logger.info(f"Polling status for video {cluster['original_order']+1}/{total_clusters}: {video_id}")
                    status = await poll_video_status(video_id, api_keys["heygen_api_key"])
                    video_url = status.get("video_url")
                    if not video_url:
                        raise ValueError(f"No video URL in status for video {video_id}")
                    
                    # Step 3: Download video
                    output_path = output_dir / f"video_cluster_{cluster['original_order']}.mp4"
                    path = await download_video(video_url, str(output_path))
                    
                    result = {
                        "path": path,
                        "order": cluster["original_order"]
                    }
                    video_downloads.append(result)
                    logger.info(f"Downloaded video {cluster['original_order']+1}/{total_clusters}: {path}")
                    
                    # Add a delay before processing the next video (if there is one)
                    if cluster["original_order"] < total_clusters - 1:
                        logger.info("Waiting 5 seconds before processing next video...")
                        await asyncio.sleep(5)
                        
                except Exception as e:
                    logger.error(f"Error processing cluster {cluster['original_order']+1}: {str(e)}")
                    raise
        else:
            # Original approach using ElevenLabs + audio upload
            logger.info("Using ElevenLabs for text-to-speech")
            
            # Generate and cluster audio
            logger.info("Generating audio from text...")
            audio_files = await generate_audio(
                config.text, 
                output_dir, 
                api_keys["elevenlabs_api_key"], 
                config.cluster_size,
                config.silence_threshold,
                config.min_silence_len,
                config.keep_silence
            )
            
            # Upload audio clusters
            logger.info("Uploading audio clusters...")
            public_urls = await upload_audio_to_public_url(audio_files)
            total_clusters = len(public_urls)
            
            # Process each audio URL fully and sequentially
            video_downloads = []
            
            for i, url in enumerate(public_urls):
                # Determine avatar pose (alternating based on reversed index)
                reversed_index = total_clusters - 1 - i
                current_avatar_id = (
                    config.front_avatar_id if reversed_index % 2 == 0 
                    else config.side_avatar_id
                )
                
                logger.info(f"Processing audio cluster {i+1}/{total_clusters}: Using avatar {current_avatar_id}")
                
                # Step 1: Create video
                logger.info(f"Creating video for audio cluster {i+1}/{total_clusters}...")
                try:
                    video_id = await create_heygen_video(
                        avatar_id=current_avatar_id,
                        avatar_style=config.avatar_style,
                        background_color=config.background_color,
                        width=config.width,
                        height=config.height,
                        api_key=api_keys["heygen_api_key"],
                        audio_url=url
                    )
                    
                    # Step 2: Poll video status
                    logger.info(f"Polling status for video {i+1}/{total_clusters}: {video_id}")
                    status = await poll_video_status(video_id, api_keys["heygen_api_key"])
                    video_url = status.get("video_url")
                    if not video_url:
                        raise ValueError(f"No video URL in status for video {video_id}")
                    
                    # Step 3: Download video
                    output_path = output_dir / f"video_cluster_{i}.mp4"
                    path = await download_video(video_url, str(output_path))
                    
                    result = {
                        "path": path,
                        "order": i
                    }
                    video_downloads.append(result)
                    logger.info(f"Downloaded video {i+1}/{total_clusters}: {path}")
                    
                    # Add a delay before processing the next video (if there is one)
                    if i < total_clusters - 1:
                        logger.info("Waiting 5 seconds before processing next video...")
                        await asyncio.sleep(5)
                        
                except Exception as e:
                    logger.error(f"Error processing audio cluster {i+1}: {str(e)}")
                    raise
        
        # Now that all videos are processed, sort video paths by order for merging
        video_paths = [item["path"] for item in sorted(video_downloads, key=lambda x: x["order"])]
        
        # Merge videos
        logger.info("Merging video segments...")
        final_path = await merge_videos(
            video_paths, 
            str(output_dir / config.output_filename)
        )

        if config.use_broll:
            try:
                logger.info("Applying B-roll post-processing...")
                from post_processing.pipeline import apply_broll_post_processing
                
                # Determine which text to use for B-roll (prefer English text if provided)
                broll_text = config.broll_text_english if config.broll_text_english else config.text
                
                # Generate a unique output filename for the B-roll version
                broll_output_filename = f"broll_{config.output_filename}"
                
                # Apply B-roll post-processing
                broll_path = await apply_broll_post_processing(
                    video_path_or_files=final_path,
                    text=broll_text,
                    output_dir=output_dir,
                    output_filename=broll_output_filename,
                    num_broll=config.broll_count,
                    broll_duration=config.broll_duration,
                    orientation=config.broll_orientation,
                    video_size=config.broll_video_size
                )
                
                # Update final path to use the B-roll version
                final_path = broll_path
                logger.info(f"B-roll post-processing complete: {final_path}")
                
            except Exception as e:
                logger.error(f"Error in B-roll post-processing: {str(e)}")
                logger.warning("Using original video without B-roll due to processing error")
            
        result = {
            "success": True,
            "final_video": final_path,
            "segments": video_paths,
            "total_clusters": total_clusters,
            "mode": "heygen_voice" if config.use_heygen_voice else "elevenlabs",
            "broll_applied": config.use_broll
        }
        
        # Add mode-specific details
        if config.use_heygen_voice:
            result["text_clusters"] = total_clusters
            
            # Add details about avatar distribution
            avatar_counts = {}
            if config.use_heygen_voice:
                # We don't have avatar_id in video_downloads anymore, so we need to count
                # based on which avatar was used for each cluster
                front_count = 0
                side_count = 0
                
                for i in range(total_clusters):
                    # Determine avatar pose using the same logic as when creating videos
                    reversed_index = total_clusters - 1 - i
                    current_avatar_id = (
                        config.front_avatar_id if reversed_index % 2 == 0 
                        else config.side_avatar_id
                    )
                    
                    if current_avatar_id == config.front_avatar_id:
                        front_count += 1
                    else:
                        side_count += 1
                
                # Add counts to result
                avatar_counts[config.front_avatar_id] = front_count
                avatar_counts[config.side_avatar_id] = side_count
                
                # Add to result
                result["avatar_distribution"] = avatar_counts
        else:
            result["audio_files"] = audio_files
        
        logger.info(f"Pipeline completed successfully. Final video: {final_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "mode": "heygen_voice" if config.use_heygen_voice else "elevenlabs"
        }
    
def _is_text_english(text: str) -> bool:
    """
    Determine if text is likely English based on character usage.
    This is a simple heuristic to check if the text contains non-English characters.
    """
    # Common letters in non-English Latin alphabets that use diacritics
    non_english_chars = set('ăâîșțéáóúíäëïöüàèìòùç')
    
    text = text.lower()
    non_english_count = sum(1 for c in text if c in non_english_chars)
    
    # If significant non-English characters present, assume non-English
    return non_english_count < len(text) * 0.01  # 1% threshold