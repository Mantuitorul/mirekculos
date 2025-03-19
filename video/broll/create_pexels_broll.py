#!/usr/bin/env python3
# create_pexels_broll.py
"""
Script to fetch broll footage from Pexels API and combine it with extracted audio.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
import subprocess
from post_processing.broll_service import BRollService
from post_processing.query_enhancer import QueryEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target dimensions - must match the HeyGen output
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280

# English translation mapping for better search results
KEYWORD_TRANSLATIONS = {
    "laboratoare": "laboratory",
    "moderne": "modern",
    "școli": "school",
    "tehnologie": "technology",
    "clasă": "classroom",
    "elevi": "students",
    "studenți": "university students",
    "profesori": "teachers",
    "antreprenori": "entrepreneurs",
    "cercetare": "research",
    "educație": "education",
    "educaționale": "educational",
    "inovație": "innovation",
    "revoluție": "revolution",
    "învățământ": "learning",
    "Arată": "show",
    "contexte": "context",
    "grupuri": "groups",
    "oameni": "people"
}

def get_video_info(video_path):
    """Get video information using FFmpeg."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFprobe error: {result.stderr}")
            return None
            
        data = json.loads(result.stdout)
        
        if not data or 'streams' not in data or not data['streams']:
            logger.error(f"No video streams found in {video_path}")
            return None
            
        stream = data['streams'][0]
        
        # If duration is not in stream, try format
        if 'duration' not in stream:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                format_data = json.loads(result.stdout)
                if 'format' in format_data and 'duration' in format_data['format']:
                    stream['duration'] = format_data['format']['duration']
        
        return {
            'width': int(stream.get('width', 0)),
            'height': int(stream.get('height', 0)),
            'duration': float(stream.get('duration', 0))
        }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return None

def combine_video_and_audio(video_path, audio_path, output_path):
    """Combine a video with an audio file using FFmpeg."""
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', video_path,  # Input video
            '-i', audio_path,  # Input audio
            '-filter_complex',
            f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2[v]',
            '-map', '[v]',  # Use the scaled video
            '-map', '1:a',  # Use the second input's audio
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
            
        logger.info(f"Successfully combined video and audio to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error combining video and audio: {e}")
        return False

def translate_to_english(keyword):
    """Translate a Romanian keyword to English for better search results"""
    if keyword in KEYWORD_TRANSLATIONS:
        return KEYWORD_TRANSLATIONS[keyword]
    return keyword

def extract_keywords(text):
    """Extract keywords from text"""
    # Simple extraction - split by spaces and remove short words
    words = text.split()
    keywords = [w.strip(',.!?:;"\'()') for w in words]
    # Filter out short words and duplicates
    keywords = list(set([k.lower() for k in keywords if len(k) > 3]))
    return keywords[:5]  # Take top 5 keywords

async def process_segment(segment, broll_service, output_dir):
    """Process a single B-roll segment."""
    try:
        segment_index = segment["index"] + 1
        logger.info(f"Processing broll segment #{segment_index}")
        
        # Extract keywords from segment text
        segment_text = segment.get("segment_text", "")
        keywords = segment.get("keywords", [])
        if not keywords and segment_text:
            keywords = extract_keywords(segment_text)
        
        # Check if we have any keywords
        if not keywords:
            logger.warning(f"No keywords found for segment {segment_index}")
            return False
            
        logger.info(f"  Original keywords: {keywords}")
        
        # Translate keywords to English
        english_keywords = [translate_to_english(kw) for kw in keywords[:5]]
        logger.info(f"  English keywords: {english_keywords}")
        
        # Use the query enhancer to get better search terms
        query_enhancer = QueryEnhancer()
        enhanced_keywords = query_enhancer.enhance_keywords(english_keywords)
        
        # Remove numbering format if present (e.g., "1. Keyword")
        enhanced_keywords = [kw.split(". ", 1)[-1] if ". " in kw else kw for kw in enhanced_keywords]
        logger.info(f"  Enhanced keywords: {enhanced_keywords}")
        
        # Create broll directory
        broll_dir = Path(output_dir) / "broll"
        broll_dir.mkdir(exist_ok=True, parents=True)
        
        # Get B-roll video from Pexels
        broll_videos = await broll_service.get_broll_for_keywords(
            keywords=enhanced_keywords,
            orientation="portrait",  # Always use portrait for HeyGen
            size="medium",
            max_results=3
        )
        
        if not broll_videos:
            logger.warning(f"No B-roll videos found for segment {segment_index}")
            return False
            
        # Select the first video
        video_data = broll_videos[0]
        video_url = video_data.get("video_files", [{}])[0].get("link")
        
        if not video_url:
            logger.warning(f"No video URL found for segment {segment_index}")
            return False
            
        # Download the video
        temp_video_path = broll_dir / f"temp_broll_{segment_index}.mp4"
        logger.info(f"Downloading B-roll video to {temp_video_path}")
        
        # Use curl to download the video
        curl_cmd = [
            'curl',
            '-L',
            '-o', str(temp_video_path),
            video_url
        ]
        
        result = subprocess.run(
            curl_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0 or not os.path.exists(temp_video_path):
            logger.error(f"Failed to download video: {result.stderr}")
            return False
            
        # Check for extracted audio
        audio_path = segment.get("extracted_audio")
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"No extracted audio found for segment {segment_index}")
            return False
            
        # Combine video and audio
        output_path = broll_dir / f"broll_segment_{segment_index}.mp4"
        logger.info(f"Combining video and audio to {output_path}")
        
        success = combine_video_and_audio(
            video_path=str(temp_video_path),
            audio_path=audio_path,
            output_path=str(output_path)
        )
        
        if not success:
            logger.error(f"Failed to combine video and audio for segment {segment_index}")
            return False
            
        # Update segment with broll video path
        segment["broll_video"] = str(output_path)
        logger.info(f"Created B-roll video for segment {segment_index}: {output_path}")
        
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        return True
    except Exception as e:
        logger.error(f"Error processing segment {segment.get('index', '?')}: {str(e)}")
        return False

async def process_all_segments(segments, broll_service, output_dir):
    """Process all B-roll segments."""
    results = []
    for segment in segments:
        success = await process_segment(segment, broll_service, output_dir)
        if success:
            results.append(segment)
            
    return results

async def main():
    """Main entry point for the script."""
    # Get API key from environment
    from utils.config import load_environment
    env_vars = load_environment()
    pexels_api_key = env_vars.get("PEXELS_API_KEY") 
    
    if not pexels_api_key:
        raise ValueError("PEXELS_API_KEY not found in environment variables")
    
    # Get path to info file
    output_dir = Path("output")
    broll_info_path = output_dir / "broll_replacement_info.json"
    
    if not broll_info_path.exists():
        logger.error(f"B-roll info file not found: {broll_info_path}")
        return
        
    with open(broll_info_path, "r") as f:
        broll_info = json.load(f)
    
    broll_segments = broll_info.get("segments", [])
    if not broll_segments:
        logger.error("No B-roll segments found in info file")
        return
    
    logger.info(f"Processing {len(broll_segments)} B-roll segments")
    
    # Create broll service
    broll_service = BRollService(api_key=pexels_api_key)
    
    # Process all segments - directly await instead of using asyncio.run()
    await process_all_segments(broll_segments, broll_service, output_dir)
    
    # Save updated info
    with open(broll_info_path, "w") as f:
        json.dump({"segments": broll_segments}, f, indent=2)
    
    logger.info("B-roll video creation complete!")

if __name__ == "__main__":
    asyncio.run(main()) 