#!/usr/bin/env python3
# simple_broll_merge.py
"""
A simpler approach to merge video segments with B-roll.
This script directly replaces segments with their B-roll versions by index,
without relying on timestamp calculations.
"""

import os
import json
import logging
from pathlib import Path
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import concatenate_videoclips

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def merge_segments_with_broll(output_dir="output", output_filename="final_broll_output.mp4"):
    """
    Merge all segments, replacing original segments with B-roll versions where available.
    
    This approach is simpler and more direct:
    1. Load the list of segments
    2. For each segment, check if a B-roll version exists
    3. If it does, use that instead of the original segment
    4. Concatenate all segments into one final video
    """
    output_dir = Path(output_dir)
    segments_dir = output_dir / "segments"
    broll_dir = output_dir / "broll"
    output_path = output_dir / output_filename
    
    # Check if segments directory exists
    if not segments_dir.exists():
        logger.error(f"Segments directory not found: {segments_dir}")
        return False
    
    # Get all segment files
    segment_files = sorted(segments_dir.glob("segment_*.mp4"), key=lambda x: int(x.stem.split("_")[1]))
    
    if not segment_files:
        logger.error("No segment files found")
        return False
    
    logger.info(f"Found {len(segment_files)} segment files")
    
    # Load B-roll info
    broll_info_path = output_dir / "broll_replacement_info.json"
    broll_segments = []
    
    if broll_info_path.exists():
        with open(broll_info_path, "r") as f:
            broll_info = json.load(f)
            broll_segments = broll_info.get("segments", [])
        
        logger.info(f"Found {len(broll_segments)} B-roll segments to replace")
    
    # Convert B-roll indices to a set for quick lookup
    broll_indices = {segment["index"] for segment in broll_segments}
    
    # Create clip list
    clips = []
    total_duration = 0
    
    for i, segment_file in enumerate(segment_files):
        # Determine if this segment has a B-roll replacement
        segment_index = int(segment_file.stem.split("_")[1])
        
        if segment_index in broll_indices:
            # Find the matching B-roll segment
            broll_path = broll_dir / f"segment_{segment_index}_broll.mp4"
            
            if broll_path.exists():
                logger.info(f"Using B-roll for segment {segment_index}: {broll_path}")
                clip = VideoFileClip(str(broll_path))
                clips.append(clip)
            else:
                logger.warning(f"B-roll not found for segment {segment_index}, using original segment")
                clip = VideoFileClip(str(segment_file))
                clips.append(clip)
        else:
            # Use the original segment
            logger.info(f"Using original segment {segment_index}: {segment_file}")
            clip = VideoFileClip(str(segment_file))
            clips.append(clip)
        
        total_duration += clip.duration
    
    # Concatenate all clips
    logger.info(f"Concatenating {len(clips)} clips with total duration of {total_duration:.2f}s")
    
    try:
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write final video
        logger.info(f"Writing final video to {output_path}")
        final_clip.write_videofile(
            str(output_path), 
            codec="libx264", 
            audio_codec="aac", 
            fps=25,
            logger=None  # Disable moviepy's own logger
        )
        
        # Close all clips
        for clip in clips:
            clip.close()
        final_clip.close()
        
        logger.info(f"Successfully created final video: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error creating final video: {e}")
        # Try to close clips
        try:
            for clip in clips:
                clip.close()
        except:
            pass
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge video segments with B-roll replacements")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--output-file", default="final_broll_output.mp4", help="Output filename")
    
    args = parser.parse_args()
    result = merge_segments_with_broll(args.output_dir, args.output_file)
    
    if result:
        logger.info(f"Successfully merged video segments with B-roll: {result}")
    else:
        logger.error("Failed to merge video segments")

if __name__ == "__main__":
    main() 