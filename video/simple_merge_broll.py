#!/usr/bin/env python3
# simple_merge_broll.py
"""
A simplified approach to merge video segments with broll footage by segment index.
No timestamps needed - just replaces the specified segments with broll footage.
"""

import os
import json
import logging
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_segment_broll(segments_dir, broll_info_path, output_dir="output"):
    """
    Create segment-specific broll videos
    
    Args:
        segments_dir: Directory containing segment videos
        broll_info_path: Path to broll info JSON
        output_dir: Base output directory
    """
    logger.info("Creating segment-specific broll videos")
    
    # Ensure output directory exists
    broll_dir = Path(output_dir) / "broll"
    broll_dir.mkdir(exist_ok=True, parents=True)
    
    # Load broll info
    with open(broll_info_path, "r") as f:
        broll_info = json.load(f)
    
    # Get broll segments
    broll_segments = broll_info.get("segments", [])
    
    if not broll_segments:
        logger.warning("No broll segments found")
        return
    
    for segment in broll_segments:
        index = segment.get("index")
        if index is None:
            logger.warning(f"Segment has no index: {segment}")
            continue
        
        # Create filename for original segment
        segment_path = os.path.join(segments_dir, f"segment_{index}.mp4")
        
        if not os.path.exists(segment_path):
            logger.warning(f"Segment file not found: {segment_path}")
            continue
        
        # Create broll path
        broll_path = os.path.join(broll_dir, f"segment_{index}_broll.mp4")
        
        # First check if a Pexels-generated broll file exists directly
        if os.path.exists(broll_path):
            logger.info(f"Using existing Pexels broll for segment {index}: {broll_path}")
            
            # Load segment and broll
            segment_clip = VideoFileClip(segment_path)
            broll_clip = VideoFileClip(broll_path)
            
            # Extract audio from segment
            audio_dir = Path(output_dir) / "extracted_audio"
            audio_dir.mkdir(exist_ok=True, parents=True)
            audio_path = os.path.join(audio_dir, f"audio_segment_{index}.mp3")
            
            # Extract audio if needed
            if not os.path.exists(audio_path):
                logger.info(f"Extracting audio from segment {index} to {audio_path}")
                segment_clip.audio.write_audiofile(audio_path, codec="libmp3lame")
            
            # Load the audio
            audio_clip = segment_clip.audio
            
            # Resize broll to match segment dimensions
            if broll_clip.size != segment_clip.size:
                logger.info(f"Resizing broll from {broll_clip.size} to {segment_clip.size}")
                broll_clip = broll_clip.resize(width=segment_clip.size[0], height=segment_clip.size[1])
            
            # Ensure broll duration matches segment duration
            if abs(broll_clip.duration - segment_clip.duration) > 0.1:
                logger.info(f"Adjusting broll duration from {broll_clip.duration}s to {segment_clip.duration}s")
                if broll_clip.duration > segment_clip.duration:
                    # Trim
                    broll_clip = broll_clip.subclipped(0, segment_clip.duration)
                else:
                    # Loop if needed
                    logger.warning(f"Broll too short ({broll_clip.duration}s < {segment_clip.duration}s), looping video")
                    from moviepy.video.fx.loop import loop
                    broll_clip = loop(broll_clip, duration=segment_clip.duration)
            
            # Apply segment audio to broll
            broll_with_audio = broll_clip.with_audio(audio_clip)
            
            # Save broll with audio
            logger.info(f"Writing broll with audio to {broll_path}")
            broll_with_audio.write_videofile(
                broll_path,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                logger=None
            )
            
            # Update segment info
            segment["broll_path"] = broll_path
            
            # Clean up
            segment_clip.close()
            broll_clip.close()
            broll_with_audio.close()
        else:
            logger.warning(f"No broll footage found for segment {index}, using original segment")
            # Just use original segment as fallback
            import shutil
            shutil.copy2(segment_path, broll_path)
            segment["broll_path"] = broll_path
    
    # Save updated broll info
    with open(broll_info_path, "w") as f:
        json.dump(broll_info, f, indent=2)
    
    logger.info("Finished creating segment-specific broll videos")

def merge_segments_with_broll(segments_dir, broll_info_path, output_path):
    """
    Merge video segments, replacing specific segments with broll footage
    
    Args:
        segments_dir: Directory containing segment videos
        broll_info_path: Path to broll info JSON
        output_path: Path to save final video
    """
    logger.info("Merging segments with broll")
    
    # Load broll info
    with open(broll_info_path, "r") as f:
        broll_info = json.load(f)
    
    # Get broll segments 
    broll_segments = broll_info.get("segments", [])
    broll_indices = [s.get("index") for s in broll_segments if s.get("index") is not None]
    
    # Find segment files
    segment_files = []
    i = 0
    while True:
        segment_path = os.path.join(segments_dir, f"segment_{i}.mp4")
        if not os.path.exists(segment_path):
            break
        segment_files.append(segment_path)
        i += 1
    
    if not segment_files:
        logger.error(f"No segment files found in {segments_dir}")
        return False
    
    logger.info(f"Found {len(segment_files)} segment files")
    logger.info(f"Broll segments to replace: {broll_indices}")
    
    # Create clip list
    clips = []
    
    for i, segment_path in enumerate(segment_files):
        # Check if this segment should be replaced with broll
        if i in broll_indices:
            logger.info(f"Replacing segment {i} with broll")
            
            # Find matching broll segment
            matching_segment = next((s for s in broll_segments if s.get("index") == i), None)
            
            if matching_segment and "broll_path" in matching_segment:
                broll_path = matching_segment["broll_path"]
                if os.path.exists(broll_path):
                    logger.info(f"Using broll: {broll_path}")
                    clip = VideoFileClip(broll_path)
                    clips.append(clip)
                    continue
            
            # Fallback to original segment if broll not found
            logger.warning(f"Broll not found for segment {i}, using original segment")
        
        # Use original segment
        logger.info(f"Using original segment: {segment_path}")
        clip = VideoFileClip(segment_path)
        clips.append(clip)
    
    # Concatenate all clips
    logger.info(f"Concatenating {len(clips)} clips")
    final_clip = concatenate_videoclips(clips, method="compose")
    
    # Write final video
    logger.info(f"Writing final video to {output_path}")
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        logger=None
    )
    
    # Clean up
    for clip in clips:
        clip.close()
    final_clip.close()
    
    logger.info(f"Successfully created final video: {output_path}")
    return True

def main():
    # Define paths
    segments_dir = "output/segments"
    broll_info_path = "output/broll_replacement_info.json"
    output_path = "output/final_output_simple.mp4"
    
    # Fetch broll videos from Pexels
    try:
        from pexels_broll_fetcher import fetch_all_broll
        logger.info("Fetching broll videos from Pexels")
        fetch_all_broll(broll_info_path, "output/broll")
    except Exception as e:
        logger.error(f"Error fetching broll videos: {e}")
    
    # Create segment-specific broll videos
    create_segment_broll(segments_dir, broll_info_path)
    
    # Merge segments with broll
    merge_segments_with_broll(segments_dir, broll_info_path, output_path)

if __name__ == "__main__":
    main() 