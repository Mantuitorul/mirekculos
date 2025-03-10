#!/usr/bin/env python3
# audio/upload.py
"""
Audio file upload utilities.
Handles uploading audio files to public URLs for HeyGen to access.
"""

import os
import logging
import asyncio
import aiohttp
from typing import List, Union

# Configure logging
logger = logging.getLogger(__name__)

async def upload_single_file(audio_path: str, session: aiohttp.ClientSession) -> str:
    """
    Upload a single audio file to 0x0.st
    
    Args:
        audio_path: Path to the audio file
        session: aiohttp ClientSession to use for the request
        
    Returns:
        Public URL for the uploaded file
    """
    logger.info(f"Uploading file: {os.path.basename(audio_path)}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at path: {audio_path}")
    
    form = aiohttp.FormData()
    form.add_field('file', open(audio_path, 'rb'))
    
    async with session.post('https://0x0.st', data=form) as response:
        if response.status != 200:
            error_text = await response.text()
            raise ValueError(f"Failed to upload file, status: {response.status}, error: {error_text}")
        
        public_url = await response.text()
        public_url = public_url.strip()
        
        if not public_url.startswith('http'):
            raise ValueError(f"Invalid URL received from 0x0.st: {public_url}")
            
        logger.info(f"Upload successful. Public URL: {public_url}")
        return public_url

async def upload_audio_to_public_url(audio_paths: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Uploads one or more audio files to 0x0.st and returns their public URLs.
    Uses asyncio for concurrent uploads.
    
    Args:
        audio_paths: Single path string or list of paths to audio files
        
    Returns:
        Single URL string or list of URLs in same order as input paths
    """
    try:
        # Handle single path case
        if isinstance(audio_paths, str):
            logger.info(f"Uploading single audio file to 0x0.st")
            async with aiohttp.ClientSession() as session:
                return await upload_single_file(audio_paths, session)
                
        # Handle list of paths case
        logger.info(f"Uploading {len(audio_paths)} audio files to 0x0.st")
        
        async with aiohttp.ClientSession() as session:
            # Use asyncio.gather to upload files concurrently
            tasks = [upload_single_file(path, session) for path in audio_paths]
            public_urls = await asyncio.gather(*tasks)
            
        logger.info(f"All {len(public_urls)} files uploaded successfully")
        return public_urls
        
    except Exception as e:
        logger.error(f"Failed to upload audio file(s): {str(e)}")
        raise