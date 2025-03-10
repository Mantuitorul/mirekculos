#!/usr/bin/env python3
# video/merger.py
"""
Video merging utilities.
Handles combining multiple video segments into a single video.
"""

import os
import logging
import asyncio
from typing import List
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips

# Configure logging
logger = logging.getLogger(__name__)

async def merge_videos(video_files: List[str], output_filename: str) -> str:
    """
    Merges a list of video files into one final video.
    
    Args:
        video_files: List of paths to video segments
        output_filename: Name of the final merged video file
        
    Returns:
        Path to the merged video file
    """
    # Run in a thread pool since moviepy operations are CPU-bound
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _merge_videos_sync, video_files, output_filename
    )

def _merge_videos_sync(video_files: List[str], output_filename: str) -> str:
    """
    Synchronous implementation of video merging.
    
    Args:
        video_files: List of paths to video segments
        output_filename: Name of the final merged video file
        
    Returns:
        Path to the merged video file
    """
    logger.info(f"Merging {len(video_files)} video files into {output_filename}")
    
    if not video_files:
        raise ValueError("No video files to merge!")
    
    logger.info(f"Video files to merge: {video_files}")
    
    clips = []
    for file in video_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Video file {file} not found!")
            
        try:
            clip = VideoFileClip(file)
            clips.append(clip)
            logger.info(f"Added clip: {file}, duration: {clip.duration}s")
        except Exception as e:
            logger.error(f"Error loading clip {file}: {str(e)}")
            # Close already loaded clips
            for loaded_clip in clips:
                loaded_clip.close()
            raise
    
    try:
        # Concatenate the video clips preserving their order
        logger.info("Creating final composition...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        logger.info(f"Writing final video to {output_filename}...")
        final_clip.write_videofile(
            output_filename, 
            codec="libx264", 
            audio_codec="aac",
            logger=None  # Disable moviepy's own logger
        )
        
        logger.info(f"Final video created successfully: {output_filename}")
        return output_filename
        
    finally:
        # Clean up and close clips
        logger.info("Cleaning up video clips...")
        for clip in clips:
            clip.close()