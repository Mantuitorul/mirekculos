#!/usr/bin/env python3
# resume_pipeline.py
"""
Resume the text-to-video pipeline from the saved segments results.
This script picks up after the HeyGen video generation step.
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import pipeline functions
from pipeline_runner import extract_broll_audio, create_pexels_broll

# Import our simpler broll merge function
from simple_broll_merge import merge_segments_with_broll

async def create_placeholder_broll_videos():
    """Create placeholder broll videos if Pexels API key is not available"""
    logger.info("Creating placeholder broll videos")
    
    # Get broll info
    output_dir = Path("output")
    broll_info_path = output_dir / "broll_replacement_info.json"
    
    if not broll_info_path.exists():
        logger.error("Broll info file not found")
        return
    
    with open(broll_info_path, "r") as f:
        broll_info = json.load(f)
    
    broll_segments = broll_info.get("segments", [])
    if not broll_segments:
        logger.info("No broll segments found")
        return
    
    logger.info(f"Processing {len(broll_segments)} broll segments")
    
    # Create broll directory if it doesn't exist
    broll_dir = output_dir / "broll"
    broll_dir.mkdir(exist_ok=True)
    
    # For each broll segment, use one of the existing HeyGen videos as a placeholder
    for segment in broll_segments:
        segment_order = segment.get("order")
        segment_path = segment.get("path")
        
        if not segment_path or not os.path.exists(segment_path):
            logger.warning(f"Segment path not found for segment {segment_order}")
            continue
        
        # Use the original video as the broll video (placeholder)
        broll_path = broll_dir / f"broll_segment_{segment_order}.mp4"
        shutil.copy2(segment_path, broll_path)
        
        # Update the segment info
        segment["broll_video"] = str(broll_path)
        segment["broll_keywords"] = ["placeholder"]
    
    # Save updated info
    with open(broll_info_path, "w") as f:
        json.dump({"segments": broll_segments}, f, indent=2)
    
    logger.info("Created placeholder broll videos")

async def resume_pipeline_from_segments():
    """
    Resume the pipeline from the HeyGen video segments that were already generated.
    Continues with:
    1. Extract audio from broll segments
    2. Create Pexels broll videos
    3. Merge segments with broll using our simplified approach
    """
    output_dir = Path("output")
    
    # Load the processing results
    processing_results_path = output_dir / "processing_results.json"
    if not os.path.exists(processing_results_path):
        raise FileNotFoundError(f"Processing results file not found: {processing_results_path}")
    
    with open(processing_results_path, "r") as f:
        segments_results = json.load(f)
    
    logger.info(f"Loaded {len(segments_results)} segments from processing results")
    
    # Step 1: Extract audio from broll segments
    logger.info("Extracting audio from broll segments")
    await extract_broll_audio(segments_results)
    
    # Step 2: Create Pexels broll videos
    logger.info("Creating Pexels broll videos")
    await create_pexels_broll(segments_results)
    
    # Step 3: Merge final video using our simplified approach
    logger.info("Merging final video with simplified approach")
    output_filename = "direct_broll_output.mp4"
    final_video = merge_segments_with_broll(str(output_dir), output_filename)
    
    if not final_video:
        logger.error("Failed to merge segments with broll")
        return None
    
    logger.info(f"Pipeline completed! Final video: {final_video}")
    return final_video

if __name__ == "__main__":
    asyncio.run(resume_pipeline_from_segments()) 