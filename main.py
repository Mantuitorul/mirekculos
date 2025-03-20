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
    parser.add_argument("--front-avatar", type=str, help="HeyGen avatar ID for front shots")
    parser.add_argument("--side-avatar", type=str, help="HeyGen avatar ID for side shots")
    parser.add_argument("--voice-id", type=str, help="HeyGen voice ID (optional)")
    parser.add_argument("--emotion", type=str, choices=["Excited", "Friendly", "Serious", "Soothing", "Broadcaster"],
                        help="HeyGen voice emotion (optional)")
    parser.add_argument("--background", type=str, default="#008000", help="Background color (hex)")
    parser.add_argument("--width", type=int, default=720, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=1280, help="Video height in pixels")
    parser.add_argument("--output", type=str, default="final_output.mp4", help="Output filename")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug-dir", type=str, default="debug_output", help="Debug output directory")
    
    # Add silence removal options
    parser.add_argument("--remove-silence", action="store_true", help="Remove silence from final video")
    parser.add_argument("--silence-threshold", type=float, default=-30, help="Silence threshold in dB (default: -30)")
    parser.add_argument("--min-silence-duration", type=float, default=0.3, help="Minimum silence duration in seconds (default: 0.3)")
    parser.add_argument("--silence-keep-ratio", type=float, default=0.2, help="Portion of silence to keep for smooth transitions (0-1, default: 0.2)")
    
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
        output_filename=parsed_args.output,
        remove_silence=parsed_args.remove_silence,
        silence_threshold=parsed_args.silence_threshold,
        min_silence_duration=parsed_args.min_silence_duration,
        silence_keep_ratio=parsed_args.silence_keep_ratio
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
        
        # Show silence removal info if enabled
        if parsed_args.remove_silence:
            print(f"\nSilence removal applied with settings:")
            print(f"  - Threshold: {parsed_args.silence_threshold} dB")
            print(f"  - Min duration: {parsed_args.min_silence_duration} sec")
            print(f"  - Keep ratio: {parsed_args.silence_keep_ratio}")
            
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
    În 2025, inovația medicală primește un impuls major! Pe 21 februarie, o nouă schemă de ajutor de stat intră în vigoare, sprijinind biotehnologiile și digitalizarea în sănătate.
    Ce înseamnă asta? Fonduri pentru IMM-uri, universități și startup-uri care dezvoltă soluții revoluționare – de la platforme digitale pentru pacienți la tehnologii de diagnostic avansate.
    Scopul? O sănătate mai accesibilă, eficientă și interconectată. România se aliniază la standardele europene, oferind suport real cercetării și inovării.
    Vrei să afli mai multe? Detalii complete sunt la un click distanță!
    """
    
    # Example avatar IDs
    FRONT_AVATAR_ID = "Raul_sitting_sofa_front_close"  # woman_prim_plan_gesturi_front
    SIDE_AVATAR_ID  = "Raul_sitting_sofa_side_close"    # woman_plan_mediu_gesturi_side
    
    # Example voice ID
    VOICE_ID = "a426f8a763824ceaad3a2eb29c68e121"
    
    # Try to parse any provided arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Handle the case where only some arguments are provided (like --remove-silence)
    # Create a new args list with default values if needed
    complete_args = list(args)  # Make a copy
    
    # Check if we need to add default values
    needs_defaults = True
    
    # If both front-avatar and side-avatar are provided, no defaults needed
    if "--front-avatar" in args and "--side-avatar" in args:
        needs_defaults = False
    
    # If text or text-file is provided, we don't need default text
    has_text = "--text" in args or "--text-file" in args
    
    # Add default values if needed
    if needs_defaults:
        print("Using default values for missing required parameters:")
        
        if "--front-avatar" not in args:
            print(f"  Front avatar: {FRONT_AVATAR_ID}")
            complete_args.extend(["--front-avatar", FRONT_AVATAR_ID])
            
        if "--side-avatar" not in args:
            print(f"  Side avatar: {SIDE_AVATAR_ID}")
            complete_args.extend(["--side-avatar", SIDE_AVATAR_ID])
            
        if "--voice-id" not in args:
            print(f"  Voice ID: {VOICE_ID}")
            complete_args.extend(["--voice-id", VOICE_ID])
            
        if "--emotion" not in args:
            print(f"  Emotion: Friendly")
            complete_args.extend(["--emotion", "Friendly"])
        
        if not has_text:
            print(f"  Using example text")
            complete_args.extend(["--text", EXAMPLE_TEXT])
        
        if "--debug" not in args:
            complete_args.append("--debug")
    
    # Run with the complete arguments
    asyncio.run(main(complete_args))