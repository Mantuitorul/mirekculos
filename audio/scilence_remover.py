#!/usr/bin/env python3
# silence_remover.py
"""
Remove silent parts from a video to make it flow better.
"""

import os
import subprocess
import logging
import argparse
from pathlib import Path
import tempfile
from moviepy import VideoFileClip, concatenate_videoclips

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_silent_segments(video_path, silence_threshold=-34, min_silence_duration=0.3):
    """
    Detect silent segments in a video using ffmpeg's silencedetect filter.
    
    Args:
        video_path: Path to the video file
        silence_threshold: Silence threshold in dB (default: -30 dB)
        min_silence_duration: Minimum silence duration in seconds (default: 0.3s)
        
    Returns:
        List of (start_time, end_time) tuples representing silent segments
    """
    logger.info(f"Detecting silent segments in {video_path}")
    
    # Use ffmpeg to detect silent segments
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-af", f"silencedetect=noise={silence_threshold}dB:d={min_silence_duration}",
        "-f", "null",
        "-"
    ]
    
    try:
        # Run ffmpeg and capture the output
        result = subprocess.run(
            command,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Parse the output to extract silent segments
        stderr_output = result.stderr
        
        # Extract silence durations
        silent_segments = []
        silence_start = None
        
        for line in stderr_output.split('\n'):
            if "silence_start" in line:
                start_time = float(line.split("silence_start: ")[1].split(" ")[0])
                silence_start = start_time
            elif "silence_end" in line and silence_start is not None:
                end_time = float(line.split("silence_end: ")[1].split(" ")[0])
                duration = float(line.split("silence_duration: ")[1])
                
                # Only add segments that meet the minimum duration
                if duration >= min_silence_duration:
                    silent_segments.append((silence_start, end_time))
                
                silence_start = None
        
        logger.info(f"Detected {len(silent_segments)} silent segments")
        return silent_segments
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ffmpeg: {e}")
        return []

def remove_silence_from_video(input_path, output_path, silence_threshold=-30, min_silence_duration=0.3, keep_ratio=0.2):
    """
    Remove silent parts from a video.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        silence_threshold: Silence threshold in dB (default: -30 dB)
        min_silence_duration: Minimum silence duration in seconds (default: 0.3s)
        keep_ratio: Ratio of silence to keep (0-1), to avoid making cuts too abrupt (default: 0.2)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing video: {input_path}")
    
    try:
        # Get silent segments
        silent_segments = get_silent_segments(input_path, silence_threshold, min_silence_duration)
        
        if not silent_segments:
            logger.info("No silent segments found or error occurred")
            return False
        
        # Load the video
        video = VideoFileClip(str(input_path))
        video_duration = video.duration
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        
        # Get non-silent segments (the parts we want to keep)
        non_silent_segments = []
        current_time = 0
        
        for i, (start, end) in enumerate(silent_segments):
            # Keep the part before the silence
            if start > current_time:
                non_silent_segments.append((current_time, start))
            
            # Keep a portion of the silence (for smoother transitions)
            silence_duration = end - start
            keep_duration = silence_duration * keep_ratio
            
            if keep_duration > 0:
                if i % 2 == 0:  # Keep beginning of silence in even segments
                    non_silent_segments.append((start, start + keep_duration))
                else:  # Keep end of silence in odd segments
                    non_silent_segments.append((end - keep_duration, end))
            
            current_time = end
        
        # Add the final part after the last silence
        if current_time < video_duration:
            non_silent_segments.append((current_time, video_duration))
        
        # Create subclips for each non-silent segment
        logger.info(f"Creating {len(non_silent_segments)} subclips")
        subclips = []
        
        for i, (start, end) in enumerate(non_silent_segments):
            # Safety check for invalid times
            if end <= start or start < 0 or end > video_duration:
                logger.warning(f"Skipping invalid segment: {start:.2f} to {end:.2f}")
                continue
                
            logger.info(f"Creating subclip {i+1}: {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)")
            subclip = video.subclipped(start, end)
            subclips.append(subclip)
        
        if not subclips:
            logger.warning("No valid subclips created")
            return False
        
        # Concatenate all subclips
        logger.info(f"Concatenating {len(subclips)} subclips")
        final_clip = concatenate_videoclips(subclips, method="compose")
        
        # Calculate time saved
        new_duration = final_clip.duration
        time_saved = video_duration - new_duration
        time_saved_percent = (time_saved / video_duration) * 100
        
        logger.info(f"Original duration: {video_duration:.2f}s")
        logger.info(f"New duration: {new_duration:.2f}s")
        logger.info(f"Time saved: {time_saved:.2f}s ({time_saved_percent:.1f}%)")
        
        # Write the final video
        logger.info(f"Writing final video to {output_path}")
        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=24,
            logger=None
        )
        
        # Clean up
        video.close()
        final_clip.close()
        for clip in subclips:
            clip.close()
        
        logger.info(f"Successfully removed silence! Output saved to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error removing silence: {e}")
        return False

def remove_silence(input_path, output_path, silence_threshold=-30, min_silence_duration=0.3, keep_ratio=0.2):
    """
    Alias for remove_silence_from_video to maintain backward compatibility.
    """
    return remove_silence_from_video(input_path, output_path, silence_threshold, min_silence_duration, keep_ratio)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Remove silent parts from a video")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("--output", "-o", help="Path to save the output video (default: input_name_no_silence.mp4)")
    parser.add_argument("--threshold", "-t", type=float, default=-30, help="Silence threshold in dB (default: -30)")
    parser.add_argument("--min-duration", "-d", type=float, default=0.3, help="Minimum silence duration in seconds (default: 0.3)")
    parser.add_argument("--keep-ratio", "-k", type=float, default=0.2, help="Ratio of silence to keep for natural transitions (0-1, default: 0.2)")
    
    args = parser.parse_args()
    
    # Set input and output paths
    input_path = Path(args.input_video)
    
    if not args.output:
        # Generate output filename if not provided
        output_filename = f"{input_path.stem}_no_silence{input_path.suffix}"
        output_path = input_path.parent / output_filename
    else:
        output_path = Path(args.output)
    
    # Remove silence
    success = remove_silence_from_video(
        input_path,
        output_path,
        silence_threshold=args.threshold,
        min_silence_duration=args.min_duration,
        keep_ratio=args.keep_ratio
    )
    
    if success:
        print(f"\nSuccess! Silent parts removed. Output saved to: {output_path}")
        print(f"You can adjust sensitivity by changing threshold (--threshold) and minimum duration (--min-duration)")
    else:
        print("\nError removing silence. Check the logs for details.")

if __name__ == "__main__":
    main()