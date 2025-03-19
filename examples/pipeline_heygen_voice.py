#!/usr/bin/env python3
"""
Entry point for the text-to-video pipeline using HeyGen's voice API.
This is a simplified wrapper around the core Pipeline class.
"""

import asyncio
import logging
from pathlib import Path

from core import Pipeline, Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def run_pipeline_flow(
    text, 
    front_avatar_id, 
    side_avatar_id, 
    heygen_voice_id=None,
    heygen_emotion=None,
    avatar_style="normal",
    background_color="#008000",
    width=720,
    height=1280,
    output_filename="final_output.mp4",
    debug_mode=False,
    debug_dir="debug_output"
):
    """
    Run the full pipeline with the following steps:
    1. ChatGPT generates frame structure (front, side, broll)
    2. HeyGen generates videos (broll as front for audio)
    3. Extract audio from broll segments
    4. Replace with Pexels footage
    5. Merge all segments
    """
    # Create the pipeline
    pipeline = Pipeline(
        width=width,
        height=height,
        output_dir=Path("output"),
        debug_mode=debug_mode,
        debug_dir=Path(debug_dir) if debug_dir else None
    )
    
    # Run the pipeline
    result = await pipeline.run(
        text=text,
        front_avatar_id=front_avatar_id,
        side_avatar_id=side_avatar_id,
        heygen_voice_id=heygen_voice_id,
        heygen_emotion=heygen_emotion,
        avatar_style=avatar_style,
        background_color=background_color,
        output_filename=output_filename
    )
    
    return result

if __name__ == "__main__":
    # Set your parameters here
    TEXT = """
        Educația românească intră într-o nouă eră: Ministerul Educației devine Ministerul Educației și Cercetării!
        Elevii, studenții, profesorii și chiar antreprenorii au acum șansa să profite de un viitor mai dinamic și orientat spre tehnologie.
        Platforma "ai aflat" este asistentul AI pentru legile din România, unde poți afla orice despre legi!
    """
    
    # Required parameters: avatar IDs for front and side poses
    FRONT_MODEL_ID = "Raul_sitting_sofa_front_close" # woman_prim_plan_gesturi_front
    SIDE_MODEL_ID = "Raul_sitting_sofa_side_close" # woman_plan_mediu_gesturi_side

    # Required: HeyGen voice ID
    HEYGEN_VOICE_ID = "a426f8a763824ceaad3a2eb29c68e121"
    
    # Optional: HeyGen voice emotion
    # Options: 'Excited', 'Friendly', 'Serious', 'Soothing', 'Broadcaster'
    # Leave as None to use default voice emotion
    HEYGEN_EMOTION = "Friendly"
    
    # Optional styling parameters
    AVATAR_STYLE = "normal"  # Options: normal, happy, sad, etc.
    BACKGROUND_COLOR = "#008000"  # Green background
    WIDTH = 720  # Video width in pixels
    HEIGHT = 1280  # Video height in pixels
    OUTPUT_FILENAME = "heygen_voice_output.mp4"
    
    # Debug options
    DEBUG_MODE = True  # Set to True to save segments to JSON file for inspection
    DEBUG_DIR = "debug_output"  # Directory to save debug files
    
    # Run the pipeline
    result = asyncio.run(
        run_pipeline_flow(
            text=TEXT,
            front_avatar_id=FRONT_MODEL_ID,
            side_avatar_id=SIDE_MODEL_ID,
            heygen_voice_id=HEYGEN_VOICE_ID,
            heygen_emotion=HEYGEN_EMOTION,
            avatar_style=AVATAR_STYLE,
            background_color=BACKGROUND_COLOR,
            width=WIDTH,
            height=HEIGHT,
            output_filename=OUTPUT_FILENAME,
            debug_mode=DEBUG_MODE,
            debug_dir=DEBUG_DIR
        )
    )
    
    # Print result summary
    print("\nPipeline result:")
    if result["success"]:
        print(f"✅ Success! Final video created: {result['final_video']}")
        print(f"Total segments: {result['total_segments']}")
        
        # Show segment counts
        print("\nSegment distribution:")
        for segment_type, count in result["segment_counts"].items():
            print(f"  - {segment_type.upper()} segments: {count}")
        
        # Show broll segments if any
        if result.get("has_broll", False):
            print("\nB-roll segments processed:")
            broll_segments = result.get("broll_segments", [])
            for i, segment in enumerate(broll_segments):
                print(f"  {i+1}. Segment #{segment['order']+1}: \"{segment.get('segment_text', '')}\"")
                if "broll_video" in segment:
                    print(f"     B-roll video: {segment['broll_video']}")
        
        # Show debug info if enabled
        if DEBUG_MODE:
            debug_path = Path(DEBUG_DIR if DEBUG_DIR else "debug") / "segments_*.json"
            print(f"\nDebug files saved to: {debug_path}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")