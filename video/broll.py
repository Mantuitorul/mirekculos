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

# Import for video processing
from moviepy.video.io.VideoFileClip import VideoFileClip

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
        max_duration: int = 8
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
            
        Returns:
            List of dictionaries with video information
        """
        broll_videos = []
        broll_dir = output_dir / "broll"
        broll_dir.mkdir(exist_ok=True, parents=True)
        
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
        audio_clip.write_audiofile(output_path, codec="libmp3lame", verbose=False, logger=None)
        
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

def combine_video_with_audio(video_path: str, audio_path: str, output_path: str, target_width: int = 720, target_height: int = 1280) -> bool:
    """
    Combine a video file with an audio file.
    
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
        # Load the video and audio
        video = VideoFileClip(video_path)
        audio = VideoFileClip(audio_path).audio
        
        # Resize the video if needed
        if video.size != (target_width, target_height):
            video = video.resize(width=target_width, height=target_height)
        
        # Combine the video and audio
        combined = video.set_audio(audio)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the combined video
        combined.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f"{output_path}_temp_audio.m4a",
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # Close the clips
        video.close()
        combined.close()
        
        return True
    except Exception as e:
        logger.error(f"Error combining video and audio: {str(e)}")
        return False