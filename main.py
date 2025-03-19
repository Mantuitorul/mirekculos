#!/usr/bin/env python3
"""
Main entry point for the video generation pipeline.
"""

import asyncio
import argparse
import logging
from pathlib import Path
import sys
from core import Pipeline, Config

async def main(args=None):
    """
    Main entry point for the video generation pipeline.
    
    Args:
        args: Command line arguments (optional, for testing)
    
    Returns:
        Result dictionary from the pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Video Generation Pipeline")
    parser.add_argument("--text", type=str, help="Input text for video generation")
    parser.add_argument("--text-file", type=str, help="Input text file for video generation")
    parser.add_argument("--front-avatar", type=str, required=True, help="HeyGen avatar ID for front shots")
    parser.add_argument("--side-avatar", type=str, required=True, help="HeyGen avatar ID for side shots")
    parser.add_argument("--voice-id", type=str, help="HeyGen voice ID (optional)")
    parser.add_argument("--emotion", type=str, choices=["Excited", "Friendly", "Serious", "Soothing", "Broadcaster"],
                        help="HeyGen voice emotion (optional)")
    parser.add_argument("--background", type=str, default="#008000", help="Background color (hex)")
    parser.add_argument("--width", type=int, default=720, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=1280, help="Video height in pixels")
    parser.add_argument("--output", type=str, default="final_output.mp4", help="Output filename")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug-dir", type=str, default="debug_output", help="Debug output directory")
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Ensure either text or text-file is provided
    if not parsed_args.text and not parsed_args.text_file:
        parser.error("Either --text or --text-file is required")
    
    # Load text from file if provided
    if parsed_args.text_file:
        try:
            with open(parsed_args.text_file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            parser.error(f"Error reading text file: {str(e)}")
    else:
        text = parsed_args.text
    
    # Create the pipeline
    pipeline = Pipeline(
        width=parsed_args.width,
        height=parsed_args.height,
        debug_mode=parsed_args.debug,
        debug_dir=parsed_args.debug_dir
    )
    
    # Run the pipeline
    result = await pipeline.run(
        text=text,
        front_avatar_id=parsed_args.front_avatar,
        side_avatar_id=parsed_args.side_avatar,
        heygen_voice_id=parsed_args.voice_id,
        heygen_emotion=parsed_args.emotion,
        background_color=parsed_args.background,
        output_filename=parsed_args.output
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
        
        # Show B-roll segments if any
        if result.get("has_broll", False):
            print("\nB-roll segments processed:")
            broll_segments = result.get("broll_segments", [])
            for i, segment in enumerate(broll_segments):
                print(f"  {i+1}. Segment #{segment['order']+1}: \"{segment.get('segment_text', '')}\"")
                if "broll_video" in segment:
                    print(f"     B-roll video: {segment['broll_video']}")
        
        # Show debug info if enabled
        if parsed_args.debug:
            debug_path = Path(parsed_args.debug_dir) / "segments_*.json"
            print(f"\nDebug files saved to: {debug_path}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    return result


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Example text for Romanian video
    EXAMPLE_TEXT = """
    Educația românească intră într-o nouă eră: Ministerul Educației devine Ministerul Educației și Cercetării!
    Elevii, studenții, profesorii și chiar antreprenorii au acum șansa să profite de un viitor mai dinamic și orientat spre tehnologie.
    Platforma "ai aflat" este asistentul AI pentru legile din România, unde poți afla orice despre legi!
    """
    
    # Example avatar IDs
    FRONT_AVATAR_ID = "Raul_sitting_sofa_front_close"  # woman_prim_plan_gesturi_front
    SIDE_AVATAR_ID = "Raul_sitting_sofa_side_close"    # woman_plan_mediu_gesturi_side
    
    # Example voice ID
    VOICE_ID = "a426f8a763824ceaad3a2eb29c68e121"
    
    # Try to parse any provided arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    
    # If no arguments provided, run with example values
    if not args:
        print("No arguments provided. Running with example values:")
        print(f"  Front avatar: {FRONT_AVATAR_ID}")
        print(f"  Side avatar: {SIDE_AVATAR_ID}")
        print(f"  Voice ID: {VOICE_ID}")
        print(f"  Emotion: Friendly")
        
        asyncio.run(main([
            "--text", EXAMPLE_TEXT,
            "--front-avatar", FRONT_AVATAR_ID,
            "--side-avatar", SIDE_AVATAR_ID,
            "--voice-id", VOICE_ID,
            "--emotion", "Friendly",
            "--debug"
        ]))
    else:
        asyncio.run(main())