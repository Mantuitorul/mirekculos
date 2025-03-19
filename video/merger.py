#!/usr/bin/env python3
"""
Video merging utilities.
Handles combining multiple video segments into a single video.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips

# Configure logging
logger = logging.getLogger(__name__)

async def merge_videos(video_files: List[str], output_path: str) -> str:
    """
    Merges a list of video files into one final video.
    
    Args:
        video_files: List of paths to video segments
        output_path: Path for the final merged video
        
    Returns:
        Path to the merged video file
    """
    # Run in a thread pool since moviepy operations are CPU-bound
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _merge_videos_sync, video_files, output_path
    )

def _merge_videos_sync(video_files: List[str], output_path: str) -> str:
    """
    Synchronous implementation of video merging.
    
    Args:
        video_files: List of paths to video segments
        output_path: Path for the final merged video
        
    Returns:
        Path to the merged video file
    """
    logger.info(f"Merging {len(video_files)} video files into {output_path}")
    
    if not video_files:
        raise ValueError("No video files to merge!")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
        
        logger.info(f"Writing final video to {output_path}...")
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            logger=None  # Disable moviepy's own logger
        )
        
        logger.info(f"Final video created successfully: {output_path}")
        return output_path
        
    finally:
        # Clean up and close clips
        logger.info("Cleaning up video clips...")
        for clip in clips:
            clip.close()

async def merge_with_broll(
    segments: List[Dict[str, Any]], 
    output_path: str,
    width: int = 720,
    height: int = 1280
) -> str:
    """
    Merge video segments, replacing broll segments with appropriate footage.
    
    Args:
        segments: List of segment information dictionaries
        output_path: Path for the final merged video
        width: Video width
        height: Video height
        
    Returns:
        Path to the merged video file
    """
    # Run in a thread pool since moviepy operations are CPU-bound
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _merge_with_broll_sync, segments, output_path, width, height
    )

def _merge_with_broll_sync(
    segments: List[Dict[str, Any]], 
    output_path: str,
    width: int = 720,
    height: int = 1280
) -> str:
    """
    Synchronous implementation of merging with B-roll.
    
    Args:
        segments: List of segment information dictionaries
        output_path: Path for the final merged video
        width: Video width
        height: Video height
        
    Returns:
        Path to the merged video file
    """
    logger.info(f"Merging {len(segments)} segments with B-roll into {output_path}")
    
    if not segments:
        raise ValueError("No segments to merge!")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort segments by order
    segments = sorted(segments, key=lambda x: x.get("order", 0))
    
    clips = []
    for segment in segments:
        segment_path = segment.get("path")
        is_broll = segment.get("is_broll", False)
        has_broll = segment.get("has_broll", False)
        broll_path = segment.get("broll_video")
        
        # Log more details about each segment for better debugging
        logger.info(f"Processing segment {segment.get('order')}: is_broll={is_broll}, has_broll={has_broll}, broll_path={broll_path}")
        
        if (is_broll or has_broll) and broll_path and os.path.exists(broll_path):
            # Check if this is a final broll_segment (already combined with audio)
            is_combined_segment = "broll_segment_" in broll_path
            
            if is_combined_segment:
                # This is a pre-combined file - use it directly without further modifications
                logger.info(f"Using pre-combined B-roll segment: {broll_path}")
                try:
                    clip = VideoFileClip(broll_path)
                    if clip.audio is None:
                        logger.warning(f"Pre-combined B-roll has no audio: {broll_path}")
                    clips.append(clip)
                    logger.info(f"Successfully added pre-combined B-roll clip for segment {segment.get('order')}")
                    continue  # Skip the more complex audio handling
                except Exception as e:
                    logger.error(f"Error loading pre-combined B-roll clip {broll_path}: {str(e)}")
                    # Will fall back to standard approach
            
            # Use B-roll video - standard method
            logger.info(f"Using B-roll for segment {segment.get('order')}: {broll_path}")
            try:
                clip = VideoFileClip(broll_path)
                # Ensure clip has the correct dimensions
                if clip.size != (width, height):
                    # Try different import paths based on MoviePy version
                    try:
                        from moviepy.video.fx.Resize import resize as resize_fx
                    except ImportError:
                        try:
                            from moviepy.video.fx.Resize import resize as resize_fx
                        except ImportError:
                            # Fall back to direct resizing if available
                            if hasattr(clip, "resize"):
                                clip = clip.resize(newsize=(width, height))
                            else:
                                logger.warning(f"Could not resize video for segment {segment.get('order')}")
                                # Continue with original size
                    else:
                        clip = clip.fx(resize_fx, newsize=(width, height))
                
                # Check if the B-roll has audio
                has_audio = clip.audio is not None
                
                # If there's no audio or an extracted audio path is provided, use the original audio
                audio_path = segment.get("audio_path")
                if (not has_audio or audio_path) and segment_path and os.path.exists(segment_path):
                    logger.info(f"Adding audio from original segment to B-roll for segment {segment.get('order')}")
                    try:
                        original_clip = VideoFileClip(segment_path)
                        if original_clip.audio is not None:
                            # Use the compatible way to set audio in MoviePy
                            from moviepy.audio.AudioClip import CompositeAudioClip
                            clip = clip.set_audio(original_clip.audio)
                            logger.info(f"Successfully added audio to B-roll for segment {segment.get('order')}")
                        else:
                            logger.warning(f"Original segment {segment.get('order')} has no audio")
                        original_clip.close()
                    except AttributeError:
                        # Fall back to using direct B-roll video if audio setting fails
                        logger.warning(f"Could not set audio using set_audio method for segment {segment.get('order')}")
                        try:
                            # Try alternative method - recreate clip with audio
                            from moviepy.audio.AudioClip import CompositeAudioClip
                            original_clip = VideoFileClip(segment_path)
                            if original_clip.audio is not None:
                                # For older MoviePy versions
                                clip = VideoFileClip(broll_path, audio=True)
                                clip.audio = original_clip.audio
                                logger.info(f"Successfully added audio using alternative method for segment {segment.get('order')}")
                            original_clip.close()
                        except Exception as e:
                            logger.error(f"Alternative audio setting method failed: {str(e)}")
                            # Continue with clip as is
                elif not has_audio:
                    logger.warning(f"B-roll segment {segment.get('order')} has no audio and no original segment found")
                
                clips.append(clip)
                logger.info(f"Successfully added B-roll clip for segment {segment.get('order')}")
            except Exception as e:
                logger.error(f"Error loading B-roll clip {broll_path}: {str(e)}")
                # Fall back to original segment
                if segment_path and os.path.exists(segment_path):
                    logger.warning(f"Falling back to original segment for {segment.get('order')}")
                    clips.append(VideoFileClip(segment_path))
                
        elif segment_path and os.path.exists(segment_path):
            # Use original segment
            logger.info(f"Using original segment: {segment_path}")
            clips.append(VideoFileClip(segment_path))
        else:
            logger.warning(f"Missing segment path for segment {segment.get('order')}")
    
    if not clips:
        raise ValueError("No clips to merge!")
    
    try:
        # Concatenate the video clips preserving their order
        logger.info("Creating final composition...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        logger.info(f"Writing final video to {output_path}...")
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            logger=None  # Disable moviepy's own logger
        )
        
        logger.info(f"Final video created successfully: {output_path}")
        return output_path
        
    finally:
        # Clean up and close clips
        logger.info("Cleaning up video clips...")
        for clip in clips:
            try:
                clip.close()
            except:
                pass