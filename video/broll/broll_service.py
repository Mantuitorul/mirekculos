#!/usr/bin/env python3
# post_processing/broll_service.py
"""
Pexels API client for fetching B-roll video content.
"""

import os
import logging
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

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
    ) -> List[str]:
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
            List of paths to downloaded videos
        """
        downloaded_videos = []
        broll_dir = output_dir / "broll"
        broll_dir.mkdir(exist_ok=True, parents=True)
        
        # Try to get videos for each keyword until we reach max_videos
        for keyword in keywords:
            if len(downloaded_videos) >= max_videos:
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
                            downloaded_videos.append(video_path)
                            break
                            
                    if len(downloaded_videos) >= max_videos:
                        break
            except Exception as e:
                logger.error(f"Error getting B-roll for '{keyword}': {str(e)}")
                continue
                
        logger.info(f"Downloaded {len(downloaded_videos)} B-roll videos for keywords: {keywords}")
        return downloaded_videos