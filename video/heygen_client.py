#!/usr/bin/env python3
# video/heygen_client.py
"""
HeyGen API client for video generation.
Handles creating videos, polling for status, and downloading completed videos.
"""

import time
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# HeyGen API endpoints
HEYGEN_BASE_URL = "https://api.heygen.com"
VIDEO_GENERATE_ENDPOINT = f"{HEYGEN_BASE_URL}/v2/video/generate"
VIDEO_STATUS_ENDPOINT = f"{HEYGEN_BASE_URL}/v1/video_status.get"

# Polling settings
MAX_POLLING_ATTEMPTS = 200
POLLING_INTERVAL = 20  # seconds

async def create_heygen_video(
    avatar_id: str,
    avatar_style: str,
    background_color: str,
    width: int,
    height: int,
    api_key: str,
    audio_url: Optional[str] = None,
    input_text: Optional[str] = None,
    voice_id: Optional[str] = None,
    emotion: Optional[str] = None
) -> str:
    """
    Create a video using the HeyGen API with either audio URL or text input.
    
    Args:
        avatar_id: ID of the avatar to use
        avatar_style: Style of the avatar
        background_color: Background color for the video
        width: Width of the video
        height: Height of the video
        api_key: HeyGen API key
        audio_url: URL to the audio file (for audio mode)
        input_text: Text to convert to speech (for text mode)
        voice_id: HeyGen voice ID to use (for text mode)
        emotion: Voice emotion to use (for text mode). 
                 Options: 'Excited', 'Friendly', 'Serious', 'Soothing', 'Broadcaster'
        
    Returns:
        Video ID for the created video
    """
    # Validate input parameters
    if audio_url and (input_text or voice_id):
        raise ValueError("Cannot specify both audio_url and text input parameters (input_text/voice_id)")
    elif not audio_url and not (input_text and voice_id):
        raise ValueError("Must specify either audio_url or both input_text and voice_id")
    
    # Configure voice based on input mode
    if audio_url:
        # Audio URL mode
        voice_config = {
            "type": "audio",
            "audio_url": audio_url
        }
        logger.info(f"Using audio URL: {audio_url}")
    else:
        # Text + voice_id mode
        voice_config = {
            "type": "text",
            "voice_id": voice_id,
            "input_text": input_text
        }
        
        # Add emotion if provided
        if emotion:
            voice_config["emotion"] = emotion
        # Log only a preview of potentially long text
        preview = input_text[:100] + ("..." if len(input_text) > 100 else "")
        logger.info(f"Using text input with voice ID: {voice_id}")
        logger.info(f"Text preview: {preview}")
    
    # Build request payload
    payload = {
        "video_inputs": [{
            "character": {
                "type": "avatar",
                "avatar_id": avatar_id,
                "avatar_style": avatar_style
            },
            "voice": voice_config,
            "background": {
                "type": "color",
                "value": background_color
            }
        }],
        "dimension": {
            "width": width,
            "height": height
        }
    }
    
    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Log detailed request information
    logger.info("Preparing HeyGen API request:")
    logger.info(f"Avatar ID: {avatar_id}")
    logger.info(f"Avatar Style: {avatar_style}")
    logger.info(f"Background Color: {background_color}")
    logger.info(f"Dimensions: {width}x{height}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(VIDEO_GENERATE_ENDPOINT, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"HeyGen API request failed: {response.status} - {error_text}")
                
            response_data = await response.json()
            
            video_id = response_data.get("data", {}).get("video_id")
            if not video_id:
                logger.error(f"No video_id in response. Full response: {response_data}")
                raise ValueError("No video_id in response")
                
            logger.info(f"Video generation initiated with ID: {video_id}")
            return video_id

async def poll_video_status(video_id: str, api_key: str) -> Dict[str, Any]:
    """
    Poll the status of a video until it is completed or failed.
    
    Args:
        video_id: ID of the video to poll
        api_key: HeyGen API key
        
    Returns:
        Video status data including video_url if completed
    """
    headers = {"X-Api-Key": api_key}
    url = f"{VIDEO_STATUS_ENDPOINT}?video_id={video_id}"
    
    async with aiohttp.ClientSession() as session:
        for attempt in range(MAX_POLLING_ATTEMPTS):
            logger.info(f"Polling attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS} for video {video_id}")
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to get video status: {response.status} - {error_text}")
                    
                response_data = await response.json()
                data = response_data.get("data", {})
                status = data.get("status")
                
                if status == "completed":
                    logger.info(f"Video {video_id} completed successfully")
                    return data
                elif status == "failed":
                    error_details = data.get("error", {})
                    logger.error(f"Video {video_id} generation failed: {error_details}")
                    raise ValueError(f"Video generation failed: {error_details.get('detail', error_details)}")
                else:
                    progress = data.get("progress", "unknown")
                    logger.info(f"Video {video_id} status: {status}, progress: {progress}%")
                
            # Wait before polling again
            await asyncio.sleep(POLLING_INTERVAL)
            
        logger.error(f"Timeout reached for video {video_id} after {MAX_POLLING_ATTEMPTS} attempts")
        raise TimeoutError("Maximum polling attempts reached")

async def download_video(video_url: str, output_path: str) -> str:
    """
    Download a video from a URL and save it to a file.
    
    Args:
        video_url: URL to download the video from
        output_path: Path to save the video to
        
    Returns:
        Path to the downloaded video
    """
    logger.info(f"Downloading video from: {video_url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(video_url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Failed to download video: {response.status} - {error_text}")
                
            with open(output_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    
    logger.info(f"Video downloaded successfully: {output_path}")
    return output_path