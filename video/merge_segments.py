#!/usr/bin/env python3
# merge_segments.py
"""
Utility to merge video segments together without any replacement.
This is a simple concatenation tool for creating the base video.
"""

import os
import logging
import asyncio
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def merge_video_segments(segment_paths, output_path):
    """
    Merge a list of video segments into a single video file.
    
    Args:
        segment_paths: List of paths to video segment files
        output_path: Path to save the merged video
        
    Returns:
        True if successful, False otherwise
    """
    # Run in a thread pool since moviepy operations are CPU-bound
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _merge_videos_sync, segment_paths, output_path)

def _merge_videos_sync(segment_paths, output_path):
    """Synchronous implementation of video merging"""
    logger.info(f"Merging {len(segment_paths)} video segments")
    
    if not segment_paths:
        logger.error("No video segments provided")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load video clips
    clips = []
    first_dimensions = None
    
    for path in segment_paths:
        if not os.path.exists(path):
            logger.warning(f"Video file not found: {path}")
            continue
            
        try:
            clip = VideoFileClip(path)
            
            # Store dimensions of first clip to resize others if needed
            if not first_dimensions:
                first_dimensions = clip.size
                logger.info(f"Using dimensions from first clip: {first_dimensions}")
            
            # Resize clip if dimensions don't match
            if clip.size != first_dimensions:
                logger.info(f"Resizing clip {path} from {clip.size} to {first_dimensions}")
                clip = clip.resize(newsize=first_dimensions)
                
            clips.append(clip)
            logger.info(f"Added clip: {path}, duration: {clip.duration:.2f}s")
        except Exception as e:
            logger.error(f"Error loading clip {path}: {str(e)}")
            # Close already loaded clips
            for loaded_clip in clips:
                loaded_clip.close()
            return False
    
    if not clips:
        logger.error("No clips were successfully loaded")
        return False
    
    try:
        # Concatenate the video clips
        logger.info("Creating final composition...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        logger.info(f"Writing final video to {output_path}...")
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            fps=25,
            logger=None  # Disable moviepy's own logger
        )
        
        logger.info(f"Final video created successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating final video: {str(e)}")
        return False
    finally:
        # Clean up and close clips
        logger.info("Cleaning up video clips...")
        for clip in clips:
            try:
                clip.close()
            except:
                pass

# Alias for compatibility with old code
merge_videos = merge_video_segments

async def main():
    """Main function to merge video segments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge video segments")
    parser.add_argument("--input-dir", type=str, help="Directory containing video segments")
    parser.add_argument("--output", type=str, default="output/merged_output.mp4", help="Output path")
    parser.add_argument("files", nargs="*", help="Video files to merge (if not using --input-dir)")
    
    args = parser.parse_args()
    
    # Get list of video files
    if args.input_dir:
        input_dir = Path(args.input_dir)
        segment_paths = sorted(str(f) for f in input_dir.glob("*.mp4"))
        logger.info(f"Found {len(segment_paths)} video segments in {input_dir}")
    elif args.files:
        segment_paths = args.files
        logger.info(f"Using {len(segment_paths)} provided video files")
    else:
        logger.error("No input files specified. Use --input-dir or provide files as arguments")
        return
    
    # Merge videos
    success = await merge_video_segments(segment_paths, args.output)
    
    if success:
        logger.info(f"Videos merged successfully to: {args.output}")
    else:
        logger.error("Failed to merge videos")

if __name__ == "__main__":
    asyncio.run(main()) 