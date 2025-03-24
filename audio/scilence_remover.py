#!/usr/bin/env python3
# silence_remover.py
"""
Remove silent parts from a video to make it flow better.
Specialized for mixed content with talking and B-roll footage.
"""

import os
import subprocess
import logging
import argparse
from pathlib import Path
import tempfile
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_audio_levels(video_path, output_path=None):
    """
    Normalize audio levels to make silence detection more consistent.
    
    Args:
        video_path: Path to the input video
        output_path: Path for normalized output (or None for temporary file)
        
    Returns:
        Path to normalized video file
    """
    if output_path is None:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"normalized_{os.path.basename(video_path)}")
    
    try:
        logger.info(f"Normalizing audio levels in {video_path}")
        
        # Use ffmpeg to normalize audio
        command = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11",  # Standard loudness normalization
            "-c:v", "copy",  # Copy video stream without re-encoding
            str(output_path)
        ]
        
        subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        logger.info(f"Audio normalized to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        return video_path  # Return original path if normalization fails

def analyze_audio_section(audio_array, start_frame, num_frames, threshold_db=-45):
    """
    Analyze a section of audio to determine if it's silent.
    
    Args:
        audio_array: NumPy array of audio samples
        start_frame: Starting frame index
        num_frames: Number of frames to analyze
        threshold_db: Silence threshold in dB
        
    Returns:
        True if the section is silent, False otherwise
    """
    if start_frame >= len(audio_array):
        return True
    
    end_frame = min(start_frame + num_frames, len(audio_array))
    section = audio_array[start_frame:end_frame]
    
    # Calculate RMS value of the section
    if len(section) == 0:
        return True
        
    if section.ndim > 1:  # Stereo or multi-channel
        section = np.mean(section, axis=1)  # Convert to mono
        
    rms = np.sqrt(np.mean(section**2))
    
    # Convert to dB (with safety for zero values)
    if rms <= 0.0000001:  # Effectively silent
        return True
        
    db = 20 * np.log10(rms)
    
    # Check if below threshold
    is_silent = db < threshold_db
    return is_silent

def find_silent_segments_advanced(video_path, silence_threshold=-38, min_silence_duration=0.25):
    """
    More advanced silence detection using direct audio analysis and FFmpeg.
    
    Args:
        video_path: Path to the video file
        silence_threshold: Silence threshold in dB
        min_silence_duration: Minimum silence duration in seconds
        
    Returns:
        List of (start_time, end_time) tuples representing silent segments
    """
    logger.info(f"Advanced silence detection for {video_path}")
    
    # Step 1: Extract audio data using MoviePy
    try:
        video = VideoFileClip(str(video_path))
        audio = video.audio
        
        if audio is None:
            logger.warning("No audio track found in video")
            return []
            
        # Get audio as numpy array
        audio_array = audio.to_soundarray()
        audio_fps = audio.fps
        
        # Step 2: Analyze audio in small windows
        window_size_sec = 0.05  # 50ms windows for precise detection
        window_size = int(window_size_sec * audio_fps)
        
        silent_windows = []
        total_windows = len(audio_array) // window_size
        
        # Check every window
        for i in range(total_windows):
            start_frame = i * window_size
            is_silent = analyze_audio_section(audio_array, start_frame, window_size, silence_threshold)
            
            if is_silent:
                start_time = start_frame / audio_fps
                end_time = (start_frame + window_size) / audio_fps
                silent_windows.append((start_time, end_time, True))  # True = is silent
            else:
                start_time = start_frame / audio_fps
                end_time = (start_frame + window_size) / audio_fps
                silent_windows.append((start_time, end_time, False))  # False = not silent
        
        # Step 3: Merge adjacent silent windows into segments
        silent_segments = []
        in_silent_segment = False
        current_start = 0
        
        for start_time, end_time, is_silent in silent_windows:
            if is_silent and not in_silent_segment:
                # Start of a new silent segment
                in_silent_segment = True
                current_start = start_time
            elif not is_silent and in_silent_segment:
                # End of a silent segment
                in_silent_segment = False
                if end_time - current_start >= min_silence_duration:
                    silent_segments.append((current_start, start_time))
        
        # Don't forget the last segment if we're still in one
        if in_silent_segment and video.duration - current_start >= min_silence_duration:
            silent_segments.append((current_start, video.duration))
        
        # Clean up
        video.close()
        
        # Step 4: Cross-check with FFmpeg for validation
        # This gives us a second opinion using a different algorithm
        ffmpeg_segments = get_silent_segments(video_path, silence_threshold, min_silence_duration)
        
        # Step 5: Merge the results from both methods
        combined_segments = merge_silence_results(silent_segments, ffmpeg_segments, min_silence_duration)
        
        logger.info(f"Detected {len(combined_segments)} silent segments with advanced method")
        return combined_segments
    
    except Exception as e:
        logger.error(f"Error in advanced silence detection: {e}")
        # Fall back to standard method
        return get_silent_segments(video_path, silence_threshold, min_silence_duration)

def merge_silence_results(method1_segments, method2_segments, min_duration):
    """
    Merge silence detection results from multiple methods.
    
    Args:
        method1_segments: Segments from first method
        method2_segments: Segments from second method
        min_duration: Minimum segment duration
        
    Returns:
        Merged list of segments
    """
    # Start with all segments from both methods
    all_segments = list(method1_segments) + list(method2_segments)
    
    # Sort by start time
    all_segments.sort(key=lambda x: x[0])
    
    # Merge overlapping segments
    if not all_segments:
        return []
        
    merged = [all_segments[0]]
    
    for current in all_segments[1:]:
        previous = merged[-1]
        
        # If current segment overlaps with previous one, merge them
        if current[0] <= previous[1]:
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            # No overlap, add as new segment
            merged.append(current)
    
    # Filter out segments that are too short
    return [seg for seg in merged if seg[1] - seg[0] >= min_duration]

def get_silent_segments(video_path, silence_threshold=-38, min_silence_duration=0.25):
    """
    Detect silent segments in a video using ffmpeg's silencedetect filter.
    
    Args:
        video_path: Path to the video file
        silence_threshold: Silence threshold in dB
        min_silence_duration: Minimum silence duration in seconds
        
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

def smooth_transitions(subclips, crossfade_duration=0.1):
    """
    Apply subtle crossfades between clips for smoother transitions.
    
    Args:
        subclips: List of video subclips
        crossfade_duration: Duration of crossfade in seconds
        
    Returns:
        List of clips with crossfades applied
    """
    if len(subclips) < 2:
        return subclips
        
    result = [subclips[0]]
    
    try:
        for i in range(1, len(subclips)):
            # Only add crossfade if both clips are long enough
            if result[-1].duration > crossfade_duration*1.5 and subclips[i].duration > crossfade_duration*1.5:
                # Add crossfade
                result[-1] = result[-1].crossfadeout(crossfade_duration)
                next_clip = subclips[i].crossfadein(crossfade_duration)
                result.append(next_clip)
            else:
                # Clips too short for crossfade, just append
                result.append(subclips[i])
        
        return result
    except Exception as e:
        logger.warning(f"Error applying crossfades: {e}")
        return subclips  # Return original subclips if crossfades fail

def remove_silence_from_video(
    input_path, 
    output_path, 
    silence_threshold=-45,      # More sensitive threshold
    min_silence_duration=0.2,   # Shorter minimum duration
    keep_ratio=0.05,            # Keep very little silence
    normalize_audio=True,       # Normalize audio before processing
    use_advanced_detection=True,# Use advanced silence detection
    apply_crossfades=True,      # Apply subtle crossfades between cuts
    ultra_aggressive=True       # Most aggressive silence removal
):
    """
    Remove silent parts from a video, specialized for mixed content with B-roll.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        silence_threshold: Silence threshold in dB
        min_silence_duration: Minimum silence duration in seconds
        keep_ratio: Ratio of silence to keep for transitions
        normalize_audio: Whether to normalize audio before processing
        use_advanced_detection: Use more advanced silence detection
        apply_crossfades: Apply subtle crossfades between cuts
        ultra_aggressive: Use the most aggressive settings
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing video: {input_path} with ultra_aggressive={ultra_aggressive}")
    
    try:
        # Step 1: Normalize audio if requested
        processing_path = input_path
        if normalize_audio:
            processing_path = normalize_audio_levels(input_path)
        
        # Step 2: Detect silent segments using the appropriate method
        if use_advanced_detection:
            silent_segments = find_silent_segments_advanced(
                processing_path, 
                silence_threshold=silence_threshold,
                min_silence_duration=min_silence_duration
            )
        else:
            silent_segments = get_silent_segments(
                processing_path,
                silence_threshold=silence_threshold,
                min_silence_duration=min_silence_duration
            )
        
        if not silent_segments:
            logger.info("No silent segments found or error occurred")
            return False
        
        # Step 3: Load the video
        video = VideoFileClip(str(input_path))
        video_duration = video.duration
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        
        # Step 4: Create non-silent segments with minimal transitions
        non_silent_segments = []
        current_time = 0
        
        for i, (start, end) in enumerate(silent_segments):
            # Add the non-silent part before this silence
            if start > current_time:
                non_silent_segments.append((current_time, start))
            
            # For ultra aggressive mode, skip most silence completely
            if ultra_aggressive:
                # Skip all short silences
                if end - start < 0.75:
                    # Skip completely
                    pass
                else:
                    # For longer silences, keep a tiny bit (1-4% based on duration)
                    duration = end - start
                    # Longer silences get slightly more kept (but still very little)
                    dynamic_ratio = min(0.01 + (duration * 0.005), 0.04)
                    keep_duration = duration * dynamic_ratio
                    
                    # Add a tiny bit at the end of the silence
                    if keep_duration > 0:
                        non_silent_segments.append((end - keep_duration, end))
            else:
                # Standard approach with keep_ratio
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
        
        # Step 5: Create subclips for each non-silent segment
        logger.info(f"Creating {len(non_silent_segments)} subclips")
        subclips = []
        
        for i, (start, end) in enumerate(non_silent_segments):
            # Safety check for invalid times
            if end <= start or start < 0 or end > video_duration:
                logger.warning(f"Skipping invalid segment: {start:.2f} to {end:.2f}")
                continue
            
            # Skip segments that are too short (under 0.1s) - they can cause issues
            if end - start < 0.1:
                logger.warning(f"Skipping too-short segment: {start:.2f} to {end:.2f}")
                continue
                
            logger.info(f"Creating subclip {i+1}: {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)")
            try:
                subclip = video.subclipped(start, end)
                subclips.append(subclip)
            except Exception as e:
                logger.error(f"Error creating subclip {i+1}: {e}")
                # Try to continue with other subclips
        
        if not subclips:
            logger.warning("No valid subclips created")
            return False
        
        # Step 6: Apply crossfades if requested
        if apply_crossfades and len(subclips) > 1:
            logger.info("Applying subtle crossfades between clips")
            crossfade_duration = 0.05 if ultra_aggressive else 0.1
            subclips = smooth_transitions(subclips, crossfade_duration)
        
        # Step 7: Concatenate all subclips
        logger.info(f"Concatenating {len(subclips)} subclips")
        final_clip = concatenate_videoclips(subclips, method="compose")
        
        # Calculate time saved
        new_duration = final_clip.duration
        time_saved = video_duration - new_duration
        time_saved_percent = (time_saved / video_duration) * 100
        
        logger.info(f"Original duration: {video_duration:.2f}s")
        logger.info(f"New duration: {new_duration:.2f}s")
        logger.info(f"Time saved: {time_saved:.2f}s ({time_saved_percent:.1f}%)")
        
        # Step 8: Write the final video
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
            try:
                clip.close()
            except:
                pass
                
        # Clean up temp file if created
        if normalize_audio and processing_path != input_path:
            try:
                os.remove(processing_path)
            except:
                pass
        
        logger.info(f"Successfully removed silence! Output saved to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error removing silence: {e}")
        return False

# Update the alias function to maintain backward compatibility
def remove_silence(input_path, output_path, silence_threshold=-45, min_silence_duration=0.2, keep_ratio=0.05):
    """
    Alias for remove_silence_from_video with ultra-aggressive defaults.
    """
    return remove_silence_from_video(
        input_path, 
        output_path, 
        silence_threshold=silence_threshold, 
        min_silence_duration=min_silence_duration, 
        keep_ratio=keep_ratio,
        ultra_aggressive=True
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Remove silent parts from a video")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("--output", "-o", help="Path to save the output video (default: input_name_no_silence.mp4)")
    parser.add_argument("--threshold", "-t", type=float, default=-45, help="Silence threshold in dB (default: -45)")
    parser.add_argument("--min-duration", "-d", type=float, default=0.2, help="Minimum silence duration in seconds (default: 0.2)")
    parser.add_argument("--keep-ratio", "-k", type=float, default=0.05, help="Ratio of silence to keep (0-1, default: 0.05)")
    parser.add_argument("--ultra", "-u", action="store_true", help="Enable ultra-aggressive mode")
    parser.add_argument("--advanced", "-a", action="store_true", help="Use advanced silence detection")
    parser.add_argument("--crossfade", "-c", action="store_true", help="Apply subtle crossfades")
    
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
        keep_ratio=args.keep_ratio,
        ultra_aggressive=args.ultra,
        use_advanced_detection=args.advanced,
        apply_crossfades=args.crossfade
    )
    
    if success:
        print(f"\nSuccess! Silent parts removed. Output saved to: {output_path}")
        print(f"You can adjust sensitivity by changing threshold (--threshold) and minimum duration (--min-duration)")
    else:
        print("\nError removing silence. Check the logs for details.")

if __name__ == "__main__":
    main()