#!/usr/bin/env python3
# merge_with_broll.py
"""
Script to merge all video segments, replacing broll segments with Pexels footage.
"""

import os
import json
import logging
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import concatenate_videoclips
import moviepy.video.fx as vfx

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def resize_video(clip, width=None, height=None):
    """Resize video to match dimensions"""
    if width and height:
        return clip.resize(newsize=(width, height))
    elif width:
        return clip.resize(width=width)
    elif height:
        return clip.resize(height=height)
    return clip

def merge_with_broll(final_video_path, broll_info_path, output_path):
    """
    Merge the main video with B-roll segments.
    
    Args:
        final_video_path: Path to the main video file
        broll_info_path: Path to the B-roll replacement info JSON
        output_path: Path to save the final video
    
    Returns:
        True if successful, False otherwise
    """
    # Check if files exist
    if not os.path.exists(final_video_path):
        logger.error(f"Final video file not found: {final_video_path}")
        return False
        
    if not os.path.exists(broll_info_path):
        logger.error(f"B-roll info file not found: {broll_info_path}")
        return False
    
    # Load B-roll replacement info
    with open(broll_info_path, "r") as f:
        replacement_info = json.load(f)
        
    # Load the main video
    logger.info(f"Loading main video: {final_video_path}")
    main_video = VideoFileClip(final_video_path)
    
    # Get main video dimensions
    main_width, main_height = main_video.size
    logger.info(f"Main video dimensions: {main_width}x{main_height}")
    
    # Get the segments
    segments = replacement_info.get("segments", [])
    if not segments:
        logger.warning("No segments found in replacement info")
        return False
        
    # Sort segments by index
    segments = sorted(segments, key=lambda x: x["index"])
    
    # Create clip list
    clips = []
    last_end_time = 0
    
    for segment in segments:
        segment_index = segment["index"] + 1  # 1-indexed for display
        
        # If we don't have start_time/end_time, skip this segment
        if "start_time" not in segment or "end_time" not in segment:
            logger.warning(f"Segment #{segment_index} missing timestamp info - skipping")
            continue
        
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        
        # Add the portion of the main video before this segment
        if start_time > last_end_time:
            logger.info(f"Adding main video from {last_end_time}s to {start_time}s")
            clip = main_video.subclip(last_end_time, start_time)
            clips.append(clip)
        
        # Check if B-roll exists
        broll_path = f"output/broll/broll_segment_{segment_index}.mp4"
        if os.path.exists(broll_path):
            logger.info(f"Adding B-roll segment #{segment_index} at {start_time}s")
            
            # Load broll video
            broll_clip = VideoFileClip(broll_path)
            
            # Ensure B-roll has the same dimensions as main video
            if broll_clip.size != (main_width, main_height):
                logger.info(f"Resizing B-roll from {broll_clip.size} to {main_width}x{main_height}")
                broll_clip = resize_video(broll_clip, width=main_width, height=main_height)
            
            # Get audio file from extracted audio if it exists
            audio_path = segment.get("extracted_audio")
            if audio_path and os.path.exists(audio_path):
                logger.info(f"Using extracted audio: {audio_path}")
                from moviepy.audio.io.AudioFileClip import AudioFileClip
                audio_clip = AudioFileClip(audio_path)
                
                # Set the audio to the broll clip
                broll_clip = broll_clip.set_audio(audio_clip)
            
            # Ensure broll duration matches segment duration
            segment_duration = end_time - start_time
            if abs(broll_clip.duration - segment_duration) > 0.1:  # Allow 0.1s difference
                logger.info(f"Adjusting B-roll duration from {broll_clip.duration}s to {segment_duration}s")
                if broll_clip.duration > segment_duration:
                    # Trim
                    broll_clip = broll_clip.subclipped(0, segment_duration)
                else:
                    # Loop if needed
                    logger.warning(f"B-roll too short ({broll_clip.duration}s < {segment_duration}s), looping video")
                    from moviepy.video.fx import loop
                    broll_clip = loop(broll_clip, duration=segment_duration)
                
            # Add the broll clip
            clips.append(broll_clip)
        else:
            logger.warning(f"B-roll segment #{segment_index} not found at {broll_path}, using main video for this segment")
            clip = main_video.subclipped(start_time, end_time)
            clips.append(clip)
            
        last_end_time = end_time
    
    # Add the remainder of the main video
    if last_end_time < main_video.duration:
        logger.info(f"Adding remaining main video from {last_end_time}s to {main_video.duration}s")
        clip = main_video.subclipped(last_end_time)
        clips.append(clip)
    
    # Concatenate all clips
    logger.info(f"Concatenating {len(clips)} clips")
    if not clips:
        logger.error("No clips to concatenate")
        return False
    
    try:
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Ensure final clip has correct dimensions
        if final_clip.size != (main_width, main_height):
            logger.info(f"Resizing final clip to {main_width}x{main_height}")
            final_clip = resize_video(final_clip, width=main_width, height=main_height)
        
        # Write final video
        logger.info(f"Writing final video to {output_path}")
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac", 
            fps=25,
            logger=None  # Disable moviepy's own logger
        )
        
        # Close all clips
        for clip in clips:
            clip.close()
        main_video.close()
        final_clip.close()
        
        logger.info(f"Successfully created final video with B-roll: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating final video: {e}")
        # Try to close clips
        try:
            for clip in clips:
                clip.close()
            main_video.close()
        except:
            pass
        return False

def main():
    # Define paths
    final_video_path = "output/heygen_voice_output.mp4"  # Output from merge_videos
    broll_info_path = "output/broll_replacement_info.json"
    output_path = "output/final_output.mp4"
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge videos
    success = merge_with_broll(final_video_path, broll_info_path, output_path)
    
    if success:
        logger.info(f"Successfully merged videos into: {output_path}")
    else:
        logger.error("Failed to merge videos")

if __name__ == "__main__":
    main() 