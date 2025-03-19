#!/usr/bin/env python3
"""
B-roll functionality for the video pipeline.
Handles fetching, processing, and integrating B-roll footage.
"""

import os
import logging
import aiohttp
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from moviepy import *
from moviepy import VideoFileClip 


# Configure logging
logger = logging.getLogger(__name__)

class BRollService:
    """Service for fetching B-roll content from Pexels API"""
    
    BASE_URL = "https://api.pexels.com/videos"
    SEARCH_ENDPOINT = f"{BASE_URL}/search"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the B-roll service with API key.
        
        Args:
            api_key: Pexels API key (defaults to PEXELS_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv("PEXELS_API_KEY")
        if not self.api_key:
            logger.warning("Pexels API key is missing. Set PEXELS_API_KEY in .env file. B-roll service will return empty results.")
    
    async def search_videos(
        self, 
        query: str, 
        orientation: str = "landscape",
        size: str = "medium",  # Full HD
        per_page: int = 10,
        min_duration: int = 3,
        max_duration: int = 10
    ) -> Dict[str, Any]:
        """
        Search for videos using the Pexels API.
        
        Args:
            query: Search keywords
            orientation: Video orientation (landscape, portrait, square)
            size: Minimum size (large=4K, medium=Full HD, small=HD)
            per_page: Number of results to return
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            
        Returns:
            Dictionary containing search results
        """
        if not self.api_key:
            logger.warning(f"Cannot search for '{query}': Pexels API key is missing")
            return {"videos": [], "total_results": 0, "page": 1, "per_page": per_page}
            
        logger.info(f"Searching for B-roll: '{query}' (orientation={orientation}, size={size})")
        
        headers = {
            "Authorization": self.api_key
        }
        
        params = {
            "query": query,
            "orientation": orientation,
            "size": size,
            "per_page": per_page
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.SEARCH_ENDPOINT, headers=headers, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Pexels API error: {response.status} - {error_text}")
                    raise RuntimeError(f"Failed to search Pexels: {response.status}")
                
                results = await response.json()
                
                # Filter by duration if needed
                if min_duration or max_duration:
                    if "videos" in results:
                        results["videos"] = [
                            v for v in results["videos"] 
                            if (min_duration <= v["duration"] <= max_duration)
                        ]
                        
                videos_count = len(results.get("videos", []))
                logger.info(f"Found {videos_count} matching B-roll videos")
                return results
    
    async def download_video(self, video: Dict[str, Any], output_dir: Path) -> str:
        """
        Download a specific video from Pexels.
        
        Args:
            video: Video object from Pexels API
            output_dir: Directory to save the video
            
        Returns:
            Path to the downloaded video file
        """
        # Ensure output directory exists
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get the video ID and prepare filename
        video_id = video["id"]
        output_path = output_dir / f"broll_{video_id}.mp4"
        
        # Find the best quality video file that matches our needs
        # Prioritize HD quality with reasonable size
        target_file = None
        
        # Sort video files by quality (hd first, then sd)
        hd_files = [f for f in video["video_files"] if f["quality"] == "hd"]
        sd_files = [f for f in video["video_files"] if f["quality"] == "sd"]
        
        # Look for 1080p HD file first
        for file in hd_files:
            if file["height"] == 1080 or (file["width"] == 1920 and file["height"] <= 1080):
                target_file = file
                break
                
        # If no 1080p, take any HD file
        if not target_file and hd_files:
            target_file = hd_files[0]
            
        # Fall back to SD if necessary
        if not target_file and sd_files:
            target_file = sd_files[0]
            
        if not target_file:
            raise ValueError(f"No suitable video files found for video ID {video_id}")
            
        # Download the video file
        video_url = target_file["link"]
        logger.info(f"Downloading B-roll: {video_id} ({target_file['width']}x{target_file['height']}, {target_file['quality']})")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to download video: {response.status} - {error_text}")
                    raise RuntimeError(f"Failed to download video {video_id}")
                    
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        
        logger.info(f"B-roll downloaded: {output_path}")
        return str(output_path)
        
    async def get_broll_for_keywords(
        self, 
        keywords: List[str], 
        output_dir: Path,
        orientation: str = "landscape",
        size: str = "medium",
        max_videos: int = 3,
        min_duration: int = 3,
        max_duration: int = 8,
        translate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get B-roll videos for a list of keywords.
        
        Args:
            keywords: List of keywords to search for
            output_dir: Directory to save videos
            orientation: Video orientation
            size: Minimum video size
            max_videos: Maximum number of videos to download
            min_duration: Minimum video duration
            max_duration: Maximum video duration
            translate: Whether to translate non-English keywords
            
        Returns:
            List of dictionaries with video information
        """
        broll_videos = []
        broll_dir = output_dir / "broll"
        broll_dir.mkdir(exist_ok=True, parents=True)
        
        # Translate keywords if needed
        if translate:
            from core.config import Config
            config = Config()
            openai_api_key = config.openai_api_key
            if openai_api_key:
                # Try to translate keywords to English
                translated_keywords = await translate_keywords_with_openai(keywords, openai_api_key)
                if translated_keywords and len(translated_keywords) > 0:
                    keywords = translated_keywords
        
        # Try to get videos for each keyword until we reach max_videos
        for keyword in keywords:
            if len(broll_videos) >= max_videos:
                break
            
            try:
                # Search for videos matching this keyword
                results = await self.search_videos(
                    query=keyword,
                    orientation=orientation,
                    size=size,
                    per_page=5,
                    min_duration=min_duration,
                    max_duration=max_duration
                )
                
                # If we found videos, download one of them
                if results.get("videos"):
                    # Get the first video that's not too long
                    for video in results["videos"]:
                        if min_duration <= video["duration"] <= max_duration:
                            video_path = await self.download_video(video, broll_dir)
                            video_info = {
                                "keyword": keyword,
                                "video_id": video["id"],
                                "path": video_path,
                                "duration": video["duration"],
                                "width": video["width"],
                                "height": video["height"],
                                "url": video["url"]
                            }
                            broll_videos.append(video_info)
                            break
                            
                    if len(broll_videos) >= max_videos:
                        break
            except Exception as e:
                logger.error(f"Error getting B-roll for '{keyword}': {str(e)}")
                continue
            
        logger.info(f"Downloaded {len(broll_videos)} B-roll videos for keywords: {keywords}")
        return broll_videos

class QueryEnhancer:
    """Enhances search queries for B-roll footage using a small OpenAI model"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo-instruct"):
        """
        Initialize the query enhancer.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Small OpenAI model to use (defaults to gpt-3.5-turbo-instruct, which is cheap)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Query enhancement will be disabled.")
        
        self.model = model
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def enhance_keywords(self, keywords: List[str], max_enhanced: int = 5) -> List[str]:
        """
        Enhance keywords for better B-roll search results.
        
        Args:
            keywords: Original keywords extracted from content
            max_enhanced: Maximum number of enhanced keywords to return
            
        Returns:
            Enhanced keywords for B-roll search
        """
        if not self.client or not keywords:
            return keywords
            
        try:
            logger.info(f"Enhancing {len(keywords)} keywords for better B-roll search")
            
            # Construct a prompt for the model
            prompt = (
                "I'm searching for B-roll footage for a video. "
                f"Based on these extracted keywords: {', '.join(keywords)}, "
                "suggest better, more specific search terms that would yield good B-roll results. "
                "Focus on visual terms that would translate well to video. "
                "Return only the search terms, one per line, no numbering or explanation. "
                f"Provide at most {max_enhanced} terms."
            )
            
            # Query the model
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            
            # Process the response
            enhanced_text = response.choices[0].text.strip()
            
            # Split by newlines and clean up
            enhanced_keywords = [k.strip() for k in enhanced_text.split('\n') if k.strip()]
            
            # Limit to max_enhanced
            enhanced_keywords = enhanced_keywords[:max_enhanced]
            
            logger.info(f"Enhanced keywords: {enhanced_keywords}")
            return enhanced_keywords
            
        except Exception as e:
            logger.error(f"Error enhancing keywords: {str(e)}")
            # Fallback to original keywords if enhancement fails
            return keywords

async def extract_audio(video_path: str, output_path: str) -> bool:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        
        logger.info(f"Extracting audio from {video_path} to {output_path}")
        video_clip = VideoFileClip(video_path)
        
        # Extract audio
        audio_clip = video_clip.audio
        
        # Write audio to file - removed verbose parameter that causes errors
        audio_clip.write_audiofile(
            output_path, 
            codec="libmp3lame",
            logger=None  # Use logger=None instead of verbose=False
        )
        
        # Close clips
        audio_clip.close()
        video_clip.close()
        
        logger.info(f"Audio extraction complete: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False

async def create_broll_segments(segments: List[Dict[str, Any]], keywords_extractor, broll_service, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Create B-roll segments for the specified segments.
    
    Args:
        segments: List of segment information dicts
        keywords_extractor: Function to extract keywords from text
        broll_service: B-roll service instance
        output_dir: Output directory for files
        
    Returns:
        Updated list of segments with B-roll information
    """
    logger.info(f"Creating B-roll segments for {len(segments)} segments")
    
    # Extract audio from each B-roll segment
    audio_dir = output_dir / "extracted_audio"
    audio_dir.mkdir(exist_ok=True, parents=True)
    
    broll_segments = []
    
    for segment in segments:
        if segment.get("is_broll", False):
            segment_index = segment["order"]
            segment_text = segment["segment_text"]
            
            # Extract audio from the segment
            if "path" in segment:
                audio_path = str(audio_dir / f"audio_segment_{segment_index}.mp3")
                if await extract_audio(segment["path"], audio_path):
                    segment["audio_path"] = audio_path
            
            # Extract keywords from the segment text and instructions
            keywords = keywords_extractor(segment_text)
            
            # Enhance keywords with query enhancer
            query_enhancer = QueryEnhancer()
            enhanced_keywords = query_enhancer.enhance_keywords(keywords)
            
            segment["keywords"] = enhanced_keywords
            segment["original_keywords"] = keywords
            
            broll_segments.append(segment)
    
    # Once all segments have audio and keywords, fetch B-roll videos
    for segment in broll_segments:
        # Get B-roll for this segment
        if "keywords" in segment:
            broll_videos = await broll_service.get_broll_for_keywords(
                keywords=segment["keywords"],
                output_dir=output_dir,
                orientation="portrait",  # Or use segment-specific orientation
                max_videos=1  # Get just one video per segment
            )
            
            if broll_videos:
                segment["broll_video"] = broll_videos[0]["path"]
                segment["broll_info"] = broll_videos[0]
            else:
                logger.warning(f"No B-roll videos found for segment {segment['order']}")
    
    # Save the updated segments
    segments_file = output_dir / "broll_segments.json"
    with open(segments_file, "w") as f:
        json.dump(broll_segments, f, indent=2)
    
    logger.info(f"Created {len(broll_segments)} B-roll segments")
    return broll_segments

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

def combine_video_with_audio(video_path: str, audio_path: str, output_path: str, target_width: int = 720, target_height: int = 1280) -> bool:
    """
    Combine a video file with an audio file using FFmpeg directly.
    Ensures that the full audio is preserved by adjusting the video length.
    
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
        from pathlib import Path
        
        logger.info(f"Combining video {video_path} with audio {audio_path}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get audio duration
        audio_duration_cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            audio_path
        ]
        
        audio_duration_process = subprocess.run(audio_duration_cmd, capture_output=True, text=True)
        if audio_duration_process.returncode != 0:
            logger.error(f"Error getting audio duration: {audio_duration_process.stderr}")
            audio_duration = None
        else:
            try:
                audio_duration = float(audio_duration_process.stdout.strip())
                logger.info(f"Audio duration: {audio_duration} seconds")
            except ValueError:
                logger.error(f"Could not parse audio duration: {audio_duration_process.stdout}")
                audio_duration = None
        
        # Get video duration
        video_duration_cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            video_path
        ]
        
        video_duration_process = subprocess.run(video_duration_cmd, capture_output=True, text=True)
        if video_duration_process.returncode != 0:
            logger.error(f"Error getting video duration: {video_duration_process.stderr}")
            video_duration = None
        else:
            try:
                video_duration = float(video_duration_process.stdout.strip())
                logger.info(f"Video duration: {video_duration} seconds")
            except ValueError:
                logger.error(f"Could not parse video duration: {video_duration_process.stdout}")
                video_duration = None
                
        logger.info(f"Using FFmpeg to combine and resize the video and audio")
        
        if audio_duration and video_duration and audio_duration > video_duration:
            # If audio is longer than video, create a looping video by concatenating with itself
            # Calculate how many times we need to loop the video
            loop_count = int(audio_duration / video_duration) + 1
            logger.info(f"Audio ({audio_duration}s) is longer than video ({video_duration}s). Creating {loop_count} loops.")
            
            # Create a temporary file for the list of videos to concatenate
            temp_concat_list = Path(os.path.dirname(output_path)) / "concat_list.txt"
            with open(temp_concat_list, 'w') as f:
                for _ in range(loop_count):
                    f.write(f"file '{os.path.basename(video_path)}'\n")
            
            # First create a looped video with correct duration
            temp_looped_video = Path(os.path.dirname(output_path)) / "temp_looped.mp4"
            
            # Copy video to the same directory as concat list for it to work properly
            temp_video_path = Path(os.path.dirname(output_path)) / os.path.basename(video_path)
            if not os.path.exists(temp_video_path):
                import shutil
                shutil.copy(video_path, temp_video_path)
            
            loop_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(temp_concat_list),
                '-c', 'copy',
                '-t', str(audio_duration),  # Trim to audio duration
                str(temp_looped_video)
            ]
            
            logger.info(f"Creating looped video: {' '.join(loop_cmd)}")
            loop_process = subprocess.run(loop_cmd, capture_output=True, text=True)
            
            if loop_process.returncode != 0:
                logger.error(f"Error creating looped video: {loop_process.stderr}")
                return False
            
            # Now combine the looped video with audio
            command = [
                'ffmpeg',
                '-y',                                    # Overwrite output files
                '-i', str(temp_looped_video),           # Looped video input
                '-i', audio_path,                        # Audio input
                '-filter_complex',                       # Use complex filter
                f"[0:v]scale={target_width}:{target_height},setsar=1[v]",  # Scale video
                '-map', '[v]',                           # Map scaled video
                '-map', '1:a',                           # Map audio from second input
                '-c:v', 'libx264',                       # Video codec
                '-c:a', 'aac',                           # Audio codec
                output_path
            ]
            
            # Run the final combination command
            logger.info(f"Running ffmpeg command: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=True)
            
            # Clean up temporary files
            if os.path.exists(str(temp_concat_list)):
                os.remove(str(temp_concat_list))
            if os.path.exists(str(temp_looped_video)):
                os.remove(str(temp_looped_video))
            if os.path.exists(str(temp_video_path)):
                os.remove(str(temp_video_path))
                
        else:
            # Standard approach, but ensuring we keep the full audio
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
                output_path
            ]
            
            # Run the standard command
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