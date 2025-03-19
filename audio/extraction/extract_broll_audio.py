#!/usr/bin/env python3
# extract_broll_audio.py
"""
Script to extract audio from segments marked as broll for later replacement with Pexels videos.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from moviepy import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_audio_with_ffmpeg(video_path, start_time, end_time, output_path):
    """
    Extract audio from a video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save the extracted audio
        
    Returns:
        True if successful, False otherwise
    """
    duration = end_time - start_time
    
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-vn',  # No video
        '-acodec', 'libmp3lame',
        '-q:a', '2',
        output_path,
        '-y'  # Overwrite if exists
    ]
    
    logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        logger.info(f"Successfully extracted audio to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return False

def extract_broll_segments(video_path, broll_segments, output_dir):
    """
    Extract audio from B-roll segments in the video.
    
    Args:
        video_path: Path to the video file
        broll_segments: List of B-roll segment information
        output_dir: Directory to save extracted audio files
    
    Returns:
        Updated broll_segments with extracted_audio paths
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    # Load the video to get its duration
    logger.info(f"Loading video to get duration: {video_path}")
    video = VideoFileClip(video_path)
    video_duration = video.duration
    logger.info(f"Video duration: {video_duration}s")
    video.close()
    
    # Create audio output directory
    audio_dir = Path(output_dir) / "extracted_audio"
    audio_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each broll segment
    for segment in broll_segments:
        index = segment["index"]
        segment_index = index + 1  # 1-indexed for display
        
        logger.info(f"Processing broll segment #{segment_index}")
        
        # If segment already has timestamps, use them
        if "start_time" in segment and "end_time" in segment:
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            
            # Ensure timestamps are within video duration
            if start_time >= video_duration:
                logger.warning(f"Start time {start_time}s is beyond video duration {video_duration}s")
                start_time = max(0, video_duration - 5)
                end_time = min(video_duration, start_time + 5)
                logger.info(f"Adjusted timestamps to {start_time}s - {end_time}s")
                
            if end_time > video_duration:
                end_time = video_duration
                logger.info(f"Adjusted end time to {end_time}s")
        else:
            # Calculate segment positions based on evenly distributed markers
            # For simplicity, we'll create 3-second segments evenly distributed
            if index == 0:
                # First segment at 20% of video
                position = 0.2
            elif index == 1:
                # Second segment at middle
                position = 0.5
            else:
                # Third segment at 80%
                position = 0.8
                
            segment_duration = 5.0  # 5 seconds per segment
            start_time = max(0, position * video_duration - segment_duration/2)
            end_time = min(video_duration, start_time + segment_duration)
            
            logger.info(f"Generated timestamps for segment #{segment_index}: {start_time:.2f}s - {end_time:.2f}s")
        
        # Update segment with timestamps
        segment["start_time"] = start_time
        segment["end_time"] = end_time
        segment["duration"] = end_time - start_time
        
        # Extract audio from this segment
        audio_path = str(audio_dir / f"audio_segment_{segment_index}.mp3")
        logger.info(f"Extracting audio for segment #{segment_index} to {audio_path}")
        
        success = extract_audio_with_ffmpeg(video_path, start_time, end_time, audio_path)
        
        if success:
            # Update segment with audio path
            segment["extracted_audio"] = audio_path
            logger.info(f"Extracted audio for segment #{segment_index}: {audio_path}")
        else:
            logger.error(f"Failed to extract audio for segment #{segment_index}")
    
    # Return the updated segments
    return broll_segments

def main():
    """Main function to extract audio from broll segments"""
    # Define paths
    video_path = "output/heygen_voice_output.mp4"  # Use the merged video as source
    output_dir = Path("output")
    broll_info_path = output_dir / "broll_replacement_info.json"
    
    # Check if the video exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Load broll info
    if not os.path.exists(broll_info_path):
        logger.error(f"B-roll info file not found: {broll_info_path}")
        return
        
    with open(broll_info_path, "r") as f:
        broll_info = json.load(f)
    
    broll_segments = broll_info.get("segments", [])
    if not broll_segments:
        logger.error("No B-roll segments found in info file")
        return
    
    logger.info(f"Processing {len(broll_segments)} B-roll segments")
    
    # Extract audio from broll segments
    updated_segments = extract_broll_segments(video_path, broll_segments, output_dir)
    
    if updated_segments:
        # Save updated info
        with open(broll_info_path, "w") as f:
            json.dump({"segments": updated_segments}, f, indent=2)
        
        logger.info(f"Updated B-roll info saved to {broll_info_path}")
        logger.info("Audio extraction complete!")
    else:
        logger.error("Failed to extract audio from B-roll segments")

if __name__ == "__main__":
    main() 