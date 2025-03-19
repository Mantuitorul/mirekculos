#!/usr/bin/env python3
"""
Script to test the pipeline with existing HeyGen videos.
This skips the HeyGen generation step and uses existing segments.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

from core.config import Config, ensure_output_dir
from text.processing import ContentAnalyzer
from video.broll import BRollService, create_broll_segments, extract_audio, combine_video_with_audio
from video.merger import merge_with_broll
from audio.processing import extract_audio_from_segments

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_pipeline(
    segments_dir: str = "output/segments",
    segments_json: str = "output/segments.json",
    output_dir: str = "output",
    output_filename: str = "test_output.mp4",
    use_broll: bool = True,
    width: int = 720,
    height: int = 1280
):
    """
    Test the pipeline with existing HeyGen videos.
    
    Args:
        segments_dir: Directory containing existing video segments
        segments_json: Path to segments JSON file
        output_dir: Output directory
        output_filename: Name of the output file
        use_broll: Whether to use B-roll processing
        width: Video width
        height: Video height
        
    Returns:
        Path to the final output video
    """
    output_path = Path(output_dir)
    ensure_output_dir(output_path)
    
    # Load the segments JSON file
    logger.info(f"Loading segments from {segments_json}")
    with open(segments_json, "r", encoding="utf-8") as f:
        segments_data = json.load(f)
    
    # Create a list of segment info with paths
    segments_info = []
    for i, segment in enumerate(segments_data):
        segment_path = os.path.join(segments_dir, f"segment_{i}.mp4")
        segment_type = segment.get("segment_shot", "front").lower()
        
        if os.path.exists(segment_path):
            segment_info = {
                "path": segment_path,
                "order": i,
                "shot_type": segment_type,
                "is_broll": segment_type == "broll",
                "segment_text": segment.get("segment_text", ""),
                "instructions": segment.get("instructions", "")
            }
            segments_info.append(segment_info)
        else:
            logger.warning(f"Segment file not found: {segment_path}")
    
    if not segments_info:
        logger.error("No valid segments found!")
        return None
    
    logger.info(f"Found {len(segments_info)} valid segments")
    
    # Step 1: Extract audio from B-roll segments if using B-roll
    if use_broll:
        broll_segments = [s for s in segments_info if s.get("is_broll", False)]
        if broll_segments:
            logger.info(f"Extracting audio from {len(broll_segments)} B-roll segments")
            extract_audio_from_segments(broll_segments, output_path)
            
            # Step 2: Generate B-roll videos
            logger.info("Creating B-roll videos")
            
            # Load config to get Pexels API key
            config = Config()
            pexels_api_key = config.pexels_api_key
            
            if pexels_api_key:
                # Create content analyzer and B-roll service
                content_analyzer = ContentAnalyzer()
                broll_service = BRollService(pexels_api_key)
                
                # Create B-roll segments
                updated_broll_segments = await create_broll_segments(
                    segments=broll_segments,
                    keywords_extractor=content_analyzer.extract_keywords,
                    broll_service=broll_service,
                    output_dir=output_path
                )
                
                # Update the segments_info with the broll information
                for updated_segment in updated_broll_segments:
                    segment_idx = updated_segment["order"]
                    for i, segment in enumerate(segments_info):
                        if segment["order"] == segment_idx:
                            # Update with B-roll video path and audio path
                            if "broll_video" in updated_segment:
                                segments_info[i]["broll_video"] = updated_segment["broll_video"]
                                segments_info[i]["has_broll"] = True
                            if "audio_path" in updated_segment:
                                segments_info[i]["audio_path"] = updated_segment["audio_path"]
                
                # Save the updated segments to processing_results.json
                with open(output_path / "processing_results.json", "w") as f:
                    json.dump(segments_info, f, indent=2)
                
                logger.info("Updated segments with B-roll information")
            else:
                logger.warning("No Pexels API key found. Skipping B-roll video generation.")
    
    # Step 3: Merge all segments into the final video
    final_output_path = output_path / output_filename
    
    logger.info(f"Merging {len(segments_info)} segments into {final_output_path}")
    
    # For each B-roll segment, make sure we combine the B-roll video with the audio
    from video.broll import combine_video_with_audio
    
    for segment in [s for s in segments_info if s.get("is_broll", False) or s.get("has_broll", False)]:
        broll_path = segment.get("broll_video")
        audio_path = segment.get("audio_path")
        segment_idx = segment.get("order")
        
        if broll_path and audio_path and os.path.exists(broll_path) and os.path.exists(audio_path):
            logger.info(f"Combining B-roll video with audio for segment {segment_idx}")
            output_video_path = output_path / "broll" / f"broll_segment_{segment_idx}.mp4"
            
            # Use the same combine_video_with_audio function from fix_broll.py
            success = combine_video_with_audio(
                broll_path,
                audio_path,
                str(output_video_path),
                width, 
                height
            )
            
            if success:
                logger.info(f"Successfully combined B-roll with audio for segment {segment_idx}")
                # Update the segment with the new combined video path
                segment["broll_video"] = str(output_video_path)
            else:
                logger.error(f"Failed to combine B-roll with audio for segment {segment_idx}")
    
    final_path = await merge_with_broll(
        segments=segments_info,
        output_path=str(final_output_path),
        width=width,
        height=height
    )
    
    logger.info(f"Pipeline test completed successfully! Final video: {final_path}")
    return final_path

# Add this code snippet to test_pipeline.py for debugging
async def debug_broll_creation(segment_index=3, output_dir="output"):
    """Debug just the B-roll creation for a specific segment"""
    from core.config import Config
    from text.processing import ContentAnalyzer
    from video.broll import BRollService, translate_keywords_with_openai
    import json
    import os
    from pathlib import Path

    config = Config()
    pexels_api_key = config.pexels_api_key
    openai_api_key = config.openai_api_key
    
    if not pexels_api_key:
        print("No Pexels API key found! Check your .env file.")
        return
    
    if not openai_api_key:
        print("Warning: No OpenAI API key found. Keyword translation won't work well.")
    
    # Load segment
    with open("output/processing_results.json", "r") as f:
        segments = json.load(f)
    
    # Find the broll segment
    segment = next((s for s in segments if s["order"] == segment_index), None)
    if not segment:
        print(f"Segment {segment_index} not found!")
        return
    
    print(f"Processing segment {segment_index}: {segment['segment_text']}")
    print(f"Is broll: {segment.get('is_broll', False)}")
    
    # Extract audio
    output_path = Path(output_dir)
    audio_dir = output_path / "extracted_audio"
    audio_dir.mkdir(exist_ok=True, parents=True)
    audio_path = audio_dir / f"audio_segment_{segment_index}.mp3"
    
    if not audio_path.exists():
        print(f"Extracting audio to {audio_path}")
        # Use our local extract_audio function
        extract_audio(segment["path"], str(audio_path))
    else:
        print(f"Audio already exists: {audio_path}")
    
    # Get content analyzer and broll service
    content_analyzer = ContentAnalyzer()
    broll_service = BRollService(pexels_api_key)
    
    # Get keywords from segment text
    keywords = content_analyzer.extract_keywords(segment['segment_text'])
    print(f"Extracted keywords: {keywords}")
    
    # Translate keywords
    translated_keywords = await translate_keywords_with_openai(keywords, openai_api_key)
    print(f"Translated keywords: {translated_keywords}")
    
    # Search Pexels for videos
    broll_videos = await broll_service.get_broll_for_keywords(
        keywords=translated_keywords,  # Use translated keywords
        output_dir=output_path,
        orientation="portrait",
        max_videos=1,
        translate=False  # Already translated
    )
    
    print(f"Found {len(broll_videos)} videos from Pexels")
    if broll_videos:
        print(f"Video path: {broll_videos[0]['path']}")
    
    # Combine video with audio
    if broll_videos and os.path.exists(audio_path):
        from video.broll import combine_video_with_audio
        output_video_path = output_path / "broll" / f"broll_segment_{segment_index}.mp4"
        print(f"Combining video and audio to {output_video_path}")
        result = combine_video_with_audio(
            broll_videos[0]['path'],
            str(audio_path),
            str(output_video_path)
        )
        print(f"Result: {result}")
        
        # Return success or failure
        return bool(result)

# Add this to the if __name__ == "__main__" block


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pipeline with existing HeyGen videos")
    parser.add_argument("--segments-dir", default="output/segments", help="Directory with video segments")
    parser.add_argument("--segments-json", default="output/segments.json", help="Path to segments JSON")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--output-file", default="test_output.mp4", help="Output filename")
    parser.add_argument("--no-broll", action="store_true", help="Skip B-roll processing")
    parser.add_argument("--remove-silence", action="store_true", help="Remove silence from final video")
    parser.add_argument("--silence-threshold", type=float, default=-30, help="Silence threshold in dB (default: -30)")
    parser.add_argument("--min-silence-duration", type=float, default=0.3, help="Minimum silence duration in seconds (default: 0.3)")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Portion of silence to keep for smooth transitions (0-1, default: 0.2)")
    parser.add_argument("--width", type=int, default=720, help="Video width")
    parser.add_argument("--height", type=int, default=1280, help="Video height")
    parser.add_argument("--debug-segment", type=int, help="Debug B-roll creation for a specific segment")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    if args.debug_segment:
        asyncio.run(debug_broll_creation(int(args.debug_segment)))
        sys.exit(0)
    
    # Run the pipeline
    final_path = asyncio.run(test_pipeline(
        segments_dir=args.segments_dir,
        segments_json=args.segments_json,
        output_dir=args.output_dir,
        output_filename=args.output_file,
        use_broll=not args.no_broll,
        width=args.width,
        height=args.height
    ))
    
    # Optionally remove silence
    if args.remove_silence and final_path:
        # Import with the correct path and handle the spelling error
        from audio.scilence_remover import remove_silence
        
        logger.info("Removing silence from final video...")
        no_silence_path = Path(final_path).parent / f"{Path(final_path).stem}_no_silence{Path(final_path).suffix}"
        
        success = remove_silence(
            final_path,
            no_silence_path,
            silence_threshold=args.silence_threshold,
            min_silence_duration=args.min_silence_duration,
            keep_ratio=args.keep_ratio
        )
        
        if success:
            logger.info(f"Successfully removed silence! Final video: {no_silence_path}")
            logger.info(f"Silence removal settings used: threshold={args.silence_threshold}dB, min_duration={args.min_silence_duration}s, keep_ratio={args.keep_ratio}")
        else:
            logger.error("Failed to remove silence")
            
        print(f"\nFinal output: {no_silence_path if success else final_path}")