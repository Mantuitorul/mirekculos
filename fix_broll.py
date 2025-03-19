#!/usr/bin/env python3
"""
Quick script to fix and test the B-roll creation functionality.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fix for MoviePy API changes
def extract_audio(video_path, audio_path):
    """Extract audio from a video file without using verbose parameter."""
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        video_clip = VideoFileClip(video_path)
        
        # Extract audio
        audio_clip = video_clip.audio
        
        # Set parameters that work with your version
        audio_clip.write_audiofile(
            audio_path, 
            codec="libmp3lame", 
            logger=None  # Use logger=None instead of verbose=False
        )
        
        # Close clips
        audio_clip.close()
        video_clip.close()
        
        logger.info(f"Audio extraction complete: {audio_path}")
        return True
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False

# Fix for MoviePy API changes (resize → resized)
def combine_video_with_audio(video_path: str, audio_path: str, output_path: str, target_width: int = 720, target_height: int = 1280) -> bool:
    """
    Combine a video file with an audio file using FFmpeg directly.
    
    Args:
        video_path: Path to the video file
        audio_path: Path to the audio file
        output_path: Path to save the combined video
        target_width: Target video width
        target_height: Target video height
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import os
        import subprocess
        
        logger.info(f"Combining video {video_path} with audio {audio_path}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Skip resizing and use FFmpeg directly
        logger.info(f"Using FFmpeg to directly combine and resize the video and audio")
        
        # Use ffmpeg to resize video and combine with audio in one command
        command = [
            'ffmpeg',
            '-y',                                    # Overwrite output files
            '-i', video_path,                        # Video input
            '-i', audio_path,                        # Audio input
            '-filter_complex',                       # Use complex filter
            f"[0:v]scale={target_width}:{target_height},setsar=1[v]",  # Scale video
            '-map', '[v]',                           # Map scaled video
            '-map', '1:a',                           # Map audio from second input
            '-c:v', 'libx264',                       # Video codec
            '-c:a', 'aac',                           # Audio codec
            '-shortest',                             # Match duration to shortest stream
            output_path
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {process.stderr}")
            return False
                    
        logger.info(f"Successfully combined video and audio: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error combining video and audio: {str(e)}")
        return False

# Translation with OpenAI
async def translate_keywords_with_openai(keywords: List[str], api_key: Optional[str] = None) -> List[str]:
    """
    Translate keywords to English and optimize them for video search using OpenAI.
    
    Args:
        keywords: List of keywords in any language
        api_key: OpenAI API key (optional)
        
    Returns:
        List of translated and optimized English keywords
    """
    openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("No OpenAI API key available for keyword translation")
        return keywords
    
    try:
        logger.info(f"Translating keywords using OpenAI: {keywords}")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Create prompt for translation and optimization
        prompt = f"""
        Translate these keywords to English and optimize them for video stock footage search.
        Focus on visual terms that would work well for video search.
        Keep each term short and specific (1-2 words if possible).
        
        Keywords: {', '.join(keywords)}
        
        Return only the translated keywords, one per line. No explanations or numbering.
        """
        
        # Use cheaper model for translation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates keywords to English and optimizes them for video search."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        # Parse response and get keywords
        text = response.choices[0].message.content.strip()
        translated_keywords = [line.strip() for line in text.split("\n") if line.strip()]
        
        logger.info(f"Translated keywords: {translated_keywords}")
        return translated_keywords
        
    except Exception as e:
        logger.error(f"Error translating keywords: {e}")
        return keywords

async def debug_broll_creation(segment_index=6, output_dir="output"):
    """Debug just the B-roll creation for a specific segment"""
    from core.config import Config
    from text.processing import ContentAnalyzer
    from video.broll import BRollService
    
    config = Config()
    pexels_api_key = config.pexels_api_key
    openai_api_key = config.openai_api_key
    
    if not pexels_api_key:
        print("No Pexels API key found! Check your .env file.")
        return False
    
    if not openai_api_key:
        print("Warning: No OpenAI API key found. Keyword translation won't work well.")
    
    # Load segment
    with open("output/processing_results.json", "r") as f:
        segments = json.load(f)
    
    # Find the broll segment
    segment = next((s for s in segments if s["order"] == segment_index), None)
    if not segment:
        print(f"Segment {segment_index} not found!")
        return False
    
    print(f"Processing segment {segment_index}: {segment['segment_text']}")
    print(f"Is broll: {segment.get('is_broll', False)}")
    
    # Extract audio
    output_path = Path(output_dir)
    audio_dir = output_path / "extracted_audio"
    audio_dir.mkdir(exist_ok=True, parents=True)
    audio_path = audio_dir / f"audio_segment_{segment_index}.mp3"
    
    if not audio_path.exists() or True:  # Force re-extract audio
        print(f"Extracting audio to {audio_path}")
        # Use our local extract_audio function
        if not extract_audio(segment["path"], str(audio_path)):
            print("Failed to extract audio!")
            return False
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
    
    # Hard-code some good search terms if translation fails
    if not translated_keywords or len(translated_keywords) == 0:
        if segment_index == 3:
            translated_keywords = ["government agency", "official building", "national agency"]
        elif segment_index == 6:
            translated_keywords = ["youth support", "teenagers", "community program"]
        else:
            translated_keywords = ["office work", "business meeting", "professional"]
    
    # Search Pexels for videos
    broll_dir = output_path / "broll"
    broll_dir.mkdir(exist_ok=True, parents=True)
    
    # Search for each keyword
    broll_video_path = None
    for keyword in translated_keywords:
        print(f"Searching Pexels for: '{keyword}'")
        
        try:
            # Search for videos matching this keyword
            url = f"https://api.pexels.com/videos/search?query={keyword}&orientation=portrait&per_page=5"
            
            # Make the request to Pexels
            import requests
            headers = {"Authorization": pexels_api_key}
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                print(f"Error searching Pexels: {response.status_code}")
                continue
                
            # Extract videos from response
            data = response.json()
            videos = data.get("videos", [])
            
            if not videos:
                print(f"No videos found for '{keyword}'")
                continue
                
            # Find a suitable video
            for video in videos:
                # Get the video file URL - prefer HD quality
                video_files = video.get("video_files", [])
                video_url = None
                
                for vf in video_files:
                    if vf.get("quality") == "hd" and vf.get("width") <= 1080:
                        video_url = vf.get("link")
                        break
                
                # If no HD quality found, use any quality
                if not video_url and video_files:
                    video_url = video_files[0].get("link")
                
                if not video_url:
                    continue
                
                # Download the video
                temp_video_path = broll_dir / f"temp_broll_{segment_index}.mp4"
                print(f"Downloading video from {video_url} to {temp_video_path}")
                
                with requests.get(video_url, stream=True) as r:
                    r.raise_for_status()
                    with open(temp_video_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                # Save the broll video path
                broll_video_path = str(temp_video_path)
                print(f"Downloaded video to {broll_video_path}")
                break
            
            # If we found a video, stop searching
            if broll_video_path:
                break
                
        except Exception as e:
            print(f"Error searching/downloading video: {e}")
    
    if not broll_video_path:
        print("Failed to find any suitable videos")
        return False
    
    # Combine video with audio
    output_video_path = broll_dir / f"broll_segment_{segment_index}.mp4"
    print(f"Combining video and audio to {output_video_path}")
    
    success = combine_video_with_audio(
        broll_video_path,
        str(audio_path),
        str(output_video_path)
    )
    
    print(f"Combining result: {success}")
    
    # If successful, update the segment with broll path
    if success:
        # Update the processing_results.json file
        segment["broll_video"] = str(output_video_path)
        segment["has_broll"] = True
        
        with open("output/processing_results.json", "w") as f:
            json.dump(segments, f, indent=2)
        
        print(f"Updated processing_results.json with broll path")
        
    return success

if __name__ == "__main__":
    # Get the segment index from command line
    segment_index = 6  # Default to segment 3
    if len(sys.argv) > 1:
        segment_index = int(sys.argv[1])
    
    print(f"Testing B-roll creation for segment {segment_index}")
    result = asyncio.run(debug_broll_creation(segment_index))
    
    if result:
        print(f"Successfully created B-roll for segment {segment_index}")
    else:
        print(f"Failed to create B-roll for segment {segment_index}")