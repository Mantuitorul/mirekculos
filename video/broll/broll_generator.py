#!/usr/bin/env python3
# broll_generator.py
"""
Generate broll videos from text segments by:
1. Translating/interpreting Romanian text to English search terms
2. Searching Pexels for relevant videos
3. Downloading videos and editing them to fit segments
4. Extracting audio from original segments
5. Combining broll visuals with segment audio
"""

import os
import json
import logging
import requests
import random
import time
import openai
from pathlib import Path
from urllib.parse import urlencode
from moviepy import VideoFileClip, AudioFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class OpenAIHelper:
    """
    Helper for generating search queries using OpenAI
    """
    def __init__(self, api_key=None):
        # Try to load API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            from utils.config import load_environment
            env_vars = load_environment()
            api_key = env_vars.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("No OpenAI API key found")
        
        self.api_key = api_key
        openai.api_key = api_key
    
    def generate_search_query(self, text, instructions=None):
        """
        Generate English search query for Pexels based on Romanian text
        
        Args:
            text: Romanian text to process
            instructions: Optional additional instructions
            
        Returns:
            English search query suitable for Pexels
        """
        try:
            # Set up the prompt for better results
            prompt = f"""
            Translate the following Romanian text to a simple English search query for finding video stock footage.
            Make the search query simple (3-5 words maximum) and focused on visual elements.
            
            Text: "{text}"
            """
            
            if instructions:
                prompt += f"\nAdditional instructions: {instructions}"
            
            # Use a cheaper model - GPT-3.5 Turbo
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates Romanian text to simple English search queries for finding stock footage videos. Keep queries simple, short, and focused on visual elements."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            # Extract the query
            search_query = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            search_query = search_query.strip('"\'')
            
            # Limit to 5 words max
            search_query = " ".join(search_query.split()[:5])
            
            logger.info(f"Generated search query: '{search_query}' from '{text}'")
            return search_query
        
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            # Fallback: just extract a few keywords
            words = text.split()
            return " ".join(words[:3])

class PexelsClient:
    """
    Client for the Pexels API to search and download videos
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

def generate_broll(segment_index, segment_text, instructions, segment_path, output_path, pexels_api_key=None, openai_api_key=None):
    """
    Generate a broll video for a segment
    
    Args:
        segment_index: Index of the segment
        segment_text: Text content of the segment
        instructions: Visual instructions for the segment
        segment_path: Path to the original segment video
        output_path: Path to save the broll video
        pexels_api_key: Pexels API key (optional, will try to load from env)
        openai_api_key: OpenAI API key (optional, will try to load from env)
        
    Returns:
        Path to the generated broll video or None if failed
    """
    logger.info(f"Generating broll for segment {segment_index}: '{segment_text}'")
    
    # Load API keys if not provided
    if not pexels_api_key:
        pexels_api_key = os.environ.get("PEXELS_API_KEY")
        
        if not pexels_api_key:
            from utils.config import load_environment
            env_vars = load_environment()
            pexels_api_key = env_vars.get("PEXELS_API_KEY")
    
    if not pexels_api_key:
        logger.error("No Pexels API key found")
        return None
    
    # Initialize helpers
    openai_helper = OpenAIHelper(openai_api_key)
    pexels_client = PexelsClient(pexels_api_key)
    
    # Step 1: Generate search query
    search_query = openai_helper.generate_search_query(segment_text, instructions)
    
    # Step 2: Search Pexels for videos
    logger.info(f"Searching Pexels for: '{search_query}' (segment {segment_index})")
    results = pexels_client.search_videos(search_query, orientation="portrait")
    
    if not results.get("videos"):
        logger.warning(f"No videos found for query: {search_query}")
        return None
    
    # Step 3: Select and download a video
    videos = results.get("videos", [])[:5]  # Get top 5
    if not videos:
        logger.warning("No suitable videos found")
        return None
    
    selected_video = random.choice(videos)
    video_url = get_best_video_file(selected_video)
    
    if not video_url:
        logger.warning("Could not get video file URL")
        return None
    
    # Create temp file for download
    temp_path = f"{output_path}.temp.mp4"
    success = pexels_client.download_video(video_url, temp_path)
    
    if not success:
        logger.error(f"Failed to download video for segment {segment_index}")
        return None
    
    # Step 4: Extract audio from original segment
    try:
        logger.info(f"Extracting audio from original segment: {segment_path}")
        segment_clip = VideoFileClip(segment_path)
        
        # Step 5: Combine broll with original audio
        logger.info(f"Combining broll with original audio")
        broll_clip = VideoFileClip(temp_path)
        
        # Resize broll to match segment dimensions
        if broll_clip.size != segment_clip.size:
            logger.info(f"Resizing broll from {broll_clip.size} to {segment_clip.size}")
            broll_clip = broll_clip.resized(width=segment_clip.size[0], height=segment_clip.size[1])
        
        # Ensure broll duration matches segment duration
        if abs(broll_clip.duration - segment_clip.duration) > 0.1:
            logger.info(f"Adjusting broll duration from {broll_clip.duration}s to {segment_clip.duration}s")
            if broll_clip.duration > segment_clip.duration:
                # Trim
                broll_clip = broll_clip.subclipped(0, segment_clip.duration)
            else:
                # Loop if needed
                logger.warning(f"Broll too short ({broll_clip.duration}s < {segment_clip.duration}s), looping video")
                from moviepy.video.fx.Loop import loop
                broll_clip = loop(broll_clip, duration=segment_clip.duration)
        
        # Apply segment audio to broll
        audio_clip = segment_clip.audio
        broll_with_audio = broll_clip.with_audio(audio_clip)
        
        # Save final broll video
        logger.info(f"Saving broll video to {output_path}")
        broll_with_audio.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            logger=None
        )
        
        # Clean up
        segment_clip.close()
        broll_clip.close()
        broll_with_audio.close()
        
        # Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        logger.info(f"Successfully generated broll for segment {segment_index}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating broll for segment {segment_index}: {e}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return None

def process_all_segments(segments_dir, broll_info_path, output_dir="output"):
    """
    Process all segments specified in the broll info file
    
    Args:
        segments_dir: Directory containing segment videos
        broll_info_path: Path to broll info JSON
        output_dir: Base output directory
        
    Returns:
        Dictionary with results
    """
    logger.info("Processing segments for broll replacement")
    
    # Ensure output directories exist
    broll_dir = Path(output_dir) / "broll"
    broll_dir.mkdir(exist_ok=True, parents=True)
    
    # Load broll info
    with open(broll_info_path, "r") as f:
        broll_info = json.load(f)
    
    # Get broll segments
    broll_segments = broll_info.get("segments", [])
    
    if not broll_segments:
        logger.warning("No broll segments found")
        return {"success": False, "message": "No broll segments found"}
    
    results = {
        "total": len(broll_segments),
        "success": 0,
        "failed": 0,
        "segments": []
    }
    
    for segment in broll_segments:
        index = segment.get("index")
        if index is None:
            logger.warning(f"Segment has no index: {segment}")
            continue
        
        segment_text = segment.get("segment_text", "")
        instructions = segment.get("instructions", "")
        
        # Create filename for original segment
        segment_path = os.path.join(segments_dir, f"segment_{index}.mp4")
        
        if not os.path.exists(segment_path):
            logger.warning(f"Segment file not found: {segment_path}")
            results["failed"] += 1
            continue
        
        # Create output path
        broll_path = os.path.join(broll_dir, f"segment_{index}_broll.mp4")
        
        # Generate broll
        output_path = generate_broll(
            segment_index=index,
            segment_text=segment_text,
            instructions=instructions,
            segment_path=segment_path,
            output_path=broll_path
        )
        
        if output_path:
            # Update segment info
            segment["broll_path"] = output_path
            segment["processed"] = True
            results["success"] += 1
            results["segments"].append({
                "index": index,
                "broll_path": output_path,
                "success": True
            })
        else:
            # Fallback: use original segment
            logger.warning(f"Failed to generate broll for segment {index}, using original segment as fallback")
            import shutil
            shutil.copy2(segment_path, broll_path)
            segment["broll_path"] = broll_path
            segment["processed"] = False
            results["failed"] += 1
            results["segments"].append({
                "index": index,
                "broll_path": broll_path,
                "success": False
            })
    
    # Save updated broll info
    with open(broll_info_path, "w") as f:
        json.dump(broll_info, f, indent=2)
    
    logger.info(f"Processed {results['total']} segments: {results['success']} successful, {results['failed']} failed")
    return results

def main():
    # Define paths
    segments_dir = "output/segments"
    broll_info_path = "output/broll_replacement_info.json"
    output_dir = "output"
    
    # Process all segments
    results = process_all_segments(segments_dir, broll_info_path, output_dir)
    
    # Print summary
    print("\nBroll Generation Summary:")
    print(f"Total segments: {results['total']}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    
    for segment in results['segments']:
        status = "✅ Success" if segment['success'] else "❌ Failed"
        print(f"{status} - Segment {segment['index']}: {segment['broll_path']}")

if __name__ == "__main__":
    main() 