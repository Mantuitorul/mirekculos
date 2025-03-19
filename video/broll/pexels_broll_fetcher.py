#!/usr/bin/env python3
# pexels_broll_fetcher.py
"""
Fetch broll videos from Pexels API based on segment text/instructions.
"""

import os
import json
import logging
import requests
import random
import time
from pathlib import Path
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PexelsClient:
    """
    Client for the Pexels API to search and download videos.
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.pexels.com/videos"
        self.headers = {
            "Authorization": api_key,
            "User-Agent": "Mozilla/5.0"
        }
    
    def search_videos(self, query, orientation="portrait", per_page=10, page=1, min_duration=3, max_duration=30):
        """
        Search for videos on Pexels
        
        Args:
            query: Search query
            orientation: Video orientation (portrait, landscape, square)
            per_page: Number of results per page
            page: Page number
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            Dictionary with search results
        """
        params = {
            "query": query,
            "orientation": orientation,
            "per_page": per_page,
            "page": page,
            "min_duration": min_duration,
            "max_duration": max_duration
        }
        
        url = f"{self.base_url}/search?{urlencode(params)}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error searching Pexels videos: {e}")
            return {"videos": []}
    
    def download_video(self, video_url, output_path):
        """
        Download a video from Pexels
        
        Args:
            video_url: URL of the video to download
            output_path: Path to save the video
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Stream the download to handle large files
            with requests.get(video_url, headers=self.headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Downloaded video to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return False

def get_best_video_file(video):
    """
    Get the best quality video file from Pexels video data
    
    Args:
        video: Pexels video data
        
    Returns:
        URL of the best quality video file
    """
    # Try to get HD quality first (around 720p)
    if "video_files" not in video:
        return None
    
    # Sort files by quality (preferring HD quality)
    quality_order = ["hd", "sd", "hls"]
    
    for quality in quality_order:
        for video_file in video["video_files"]:
            if video_file.get("quality") == quality:
                return video_file["link"]
    
    # If no preferred quality found, just return the first file
    if video["video_files"]:
        return video["video_files"][0]["link"]
    
    return None

def extract_keywords(text, instructions):
    """
    Extract search keywords from text and instructions
    
    Args:
        text: Segment text
        instructions: Segment instructions
        
    Returns:
        List of search keywords
    """
    if not instructions:
        # If no instructions, use text words as keywords
        return [word for word in text.split() if len(word) > 3]
    
    # Use instructions for better keywords
    # Remove common words and punctuation
    common_words = ["and", "the", "with", "for", "sau", "din", "şi"]
    keywords = []
    
    for word in instructions.split():
        # Clean word
        word = word.strip(",.!?;:()[]{}").lower()
        if len(word) > 3 and word not in common_words:
            keywords.append(word)
    
    return keywords[:5]  # Limit to 5 keywords

def fetch_broll_for_segment(pexels_client, segment, output_dir):
    """
    Fetch broll videos for a segment
    
    Args:
        pexels_client: PexelsClient instance
        segment: Segment data
        output_dir: Output directory
        
    Returns:
        Path to downloaded broll video or None
    """
    # Get segment text and instructions
    segment_text = segment.get("segment_text", "")
    segment_instructions = segment.get("instructions", "")
    segment_index = segment.get("index")
    
    # Extract keywords
    keywords = extract_keywords(segment_text, segment_instructions)
    
    if not keywords:
        logger.warning(f"No keywords found for segment {segment_index}")
        return None
    
    # Build search query from keywords
    search_query = " ".join(keywords[:3])  # Use first 3 keywords
    logger.info(f"Searching Pexels for: '{search_query}' (segment {segment_index})")
    
    # Search Pexels
    results = pexels_client.search_videos(search_query, orientation="portrait")
    
    if not results.get("videos"):
        logger.warning(f"No videos found for query: {search_query}")
        return None
    
    # Get a random video from results (top 5)
    videos = results.get("videos", [])[:5]
    if not videos:
        return None
    
    selected_video = random.choice(videos)
    
    # Get video file URL
    video_url = get_best_video_file(selected_video)
    if not video_url:
        logger.warning(f"No video file found for selected video")
        return None
    
    # Prepare output path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"segment_{segment_index}_broll.mp4")
    
    # Download the video
    success = pexels_client.download_video(video_url, output_path)
    
    if success:
        # Add Pexels info to segment
        segment["pexels_video_id"] = selected_video.get("id")
        segment["pexels_video_url"] = selected_video.get("url")
        segment["pexels_search_query"] = search_query
        segment["broll_path"] = output_path
        segment["broll_keywords"] = keywords
        
        return output_path
    
    return None

def fetch_all_broll(broll_info_path, output_dir="output/broll", api_key=None):
    """
    Fetch broll videos for all segments in the broll info
    
    Args:
        broll_info_path: Path to broll info JSON
        output_dir: Output directory
        api_key: Pexels API key (optional, will try to load from env)
        
    Returns:
        True if successful, False otherwise
    """
    # Load Pexels API key
    if not api_key:
        # Try to load from environment
        api_key = os.environ.get("PEXELS_API_KEY")
        
        if not api_key:
            from utils.config import load_environment
            env_vars = load_environment()
            api_key = env_vars.get("PEXELS_API_KEY")
    
    if not api_key:
        logger.error("No Pexels API key found")
        return False
    
    # Load broll info
    with open(broll_info_path, "r") as f:
        broll_info = json.load(f)
    
    # Get broll segments
    broll_segments = broll_info.get("segments", [])
    
    if not broll_segments:
        logger.warning("No broll segments found")
        return False
    
    # Initialize Pexels client
    pexels_client = PexelsClient(api_key)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch broll for each segment
    for segment in broll_segments:
        # Check if already has broll path
        if "pexels_video_id" in segment:
            logger.info(f"Segment {segment.get('index')} already has Pexels video")
            continue
        
        # Add small delay to avoid hitting API rate limits
        time.sleep(1)
        
        # Fetch broll
        broll_path = fetch_broll_for_segment(pexels_client, segment, output_dir)
        
        if broll_path:
            logger.info(f"Downloaded broll for segment {segment.get('index')}: {broll_path}")
        else:
            logger.warning(f"Failed to download broll for segment {segment.get('index')}")
    
    # Save updated broll info
    with open(broll_info_path, "w") as f:
        json.dump(broll_info, f, indent=2)
    
    logger.info("Finished fetching broll videos")
    return True

def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch broll videos from Pexels")
    parser.add_argument("--broll-info", dest="broll_info", default="output/broll_replacement_info.json",
                      help="Path to broll info JSON")
    parser.add_argument("--output-dir", dest="output_dir", default="output/broll",
                      help="Output directory for broll videos")
    parser.add_argument("--api-key", dest="api_key", default=None,
                      help="Pexels API key (optional, will try to load from env)")
    
    args = parser.parse_args()
    
    fetch_all_broll(args.broll_info, args.output_dir, args.api_key)

if __name__ == "__main__":
    main() 