#!/usr/bin/env python3
# post_processing/pipeline.py
"""
Post-processing pipeline for B-roll insertion.
Main orchestration module that ties together content analysis, B-roll retrieval, and video processing.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from moviepy.video.io.VideoFileClip import VideoFileClip
from .content_analyzer import ContentAnalyzer
from .broll_service import BRollService
from .video_processor import VideoProcessor

# Configure logging
logger = logging.getLogger(__name__)

class PostProcessingPipeline:
    """Orchestrates the B-roll insertion post-processing workflow"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the post-processing pipeline.
        
        Args:
            api_key: Pexels API key (defaults to PEXELS_API_KEY environment variable)
        """
        self.content_analyzer = ContentAnalyzer()
        self.broll_service = BRollService(api_key)
        self.video_processor = VideoProcessor()
        
    async def process(
        self,
        text: str,
        video_path: str,
        output_path: str,
        output_dir: Path,
        num_broll: int = 2,
        broll_duration: float = 5.0,
        orientation: str = "landscape",
        video_size: str = "medium",
        max_keywords: int = 5,
        fixed_interval: float = 7.0  # New parameter
    ) -> str:
        """
        Process a single video with B-roll.
        
        Args:
            text: Original text content for keyword extraction
            video_path: Path to the main video
            output_path: Path for the output video
            output_dir: Directory for intermediate files
            num_broll: Maximum number of B-roll segments to insert
            broll_duration: Duration of each B-roll segment
            orientation: Video orientation (landscape, portrait, square)
            video_size: Minimum video size (large=4K, medium=Full HD, small=HD)
            max_keywords: Maximum number of keywords to extract
            fixed_interval: Fixed interval in seconds for B-roll insertion
            
        Returns:
            Path to the processed video
        """
        logger.info(f"Starting B-roll post-processing for video: {video_path}")
        
        # Step 1: Extract keywords from text
        keywords = self.content_analyzer.extract_keywords(text, max_keywords)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Estimate video duration to calculate number of B-roll clips needed
        video_duration = 0
        try:
            with VideoFileClip(video_path) as clip:
                video_duration = clip.duration
        except Exception as e:
            logger.error(f"Error getting video duration: {str(e)}")
            # Default to a reasonable duration if we can't get the actual length
            video_duration = 60  # 1 minute as fallback
            
        # Calculate estimated number of B-roll clips needed
        edge_buffer = 5.0
        usable_duration = max(0, video_duration - (2 * edge_buffer))
        estimated_broll_count = max(1, int(usable_duration / fixed_interval))
        
        # Use the minimum of the estimated count and the user-specified num_broll
        broll_limit = min(estimated_broll_count, num_broll) if num_broll > 0 else estimated_broll_count
        
        logger.info(f"Video duration: {video_duration:.2f}s, estimated B-roll clips needed: {estimated_broll_count}")
        logger.info(f"Using B-roll limit: {broll_limit}")
        
        # Step 2: Search and download B-roll for these keywords
        broll_paths = await self.broll_service.get_broll_for_keywords(
            keywords=keywords,
            output_dir=output_dir,
            orientation=orientation,
            size=video_size,
            max_videos=broll_limit,  # Use our calculated limit
            min_duration=broll_duration,
            max_duration=broll_duration * 2
        )
        
        if not broll_paths:
            logger.warning("No B-roll videos found, returning original video")
            return video_path
                
        # Step 3: Determine insertion points in the video
        dummy_files = [video_path]
        insertion_points = self.content_analyzer.determine_broll_points(
            video_files=dummy_files,
            num_points=len(broll_paths),  # Limit to the number of B-roll videos we have
            edge_buffer=5.0,
            fixed_interval=fixed_interval  # Use our fixed interval
        )
        
        # Step 4: Insert B-roll at determined points
        processed_path = await self.video_processor.insert_broll(
            video_path=video_path,
            broll_paths=broll_paths,
            insertion_points=insertion_points,
            output_path=output_path,
            broll_duration=broll_duration
        )
        
        logger.info(f"B-roll post-processing complete: {processed_path}")
        return processed_path
        
    async def process_multifile(
        self,
        text: str,
        video_files: List[str],
        output_path: str,
        output_dir: Path,
        num_broll: int = 2,
        broll_duration: float = 5.0,
        orientation: str = "landscape",
        video_size: str = "medium",
        max_keywords: int = 5,
        fixed_interval: float = 7.0  # New parameter
    ) -> str:
        """
        Process multiple video files with B-roll.
        
        Args:
            text: Original text content for keyword extraction
            video_files: Paths to the video segments
            output_path: Path for the output video
            output_dir: Directory for intermediate files
            num_broll: Maximum number of B-roll segments to insert
            broll_duration: Duration of each B-roll segment
            orientation: Video orientation (landscape, portrait, square)
            video_size: Minimum video size (large=4K, medium=Full HD, small=HD)
            max_keywords: Maximum number of keywords to extract
            fixed_interval: Fixed interval in seconds for B-roll insertion
            
        Returns:
            Path to the processed video
        """
        logger.info(f"Starting B-roll post-processing for {len(video_files)} video files")
        
        # Step 1: Extract keywords from text
        keywords = self.content_analyzer.extract_keywords(text, max_keywords)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Estimate total video duration to calculate number of B-roll clips needed
        total_duration = 0
        for video_path in video_files:
            try:
                with VideoFileClip(video_path) as clip:
                    total_duration += clip.duration
            except Exception as e:
                logger.error(f"Error getting video duration for {video_path}: {str(e)}")
                # Skip this file for duration calculation
        
        # If we couldn't get any duration info, use a reasonable fallback
        if total_duration == 0:
            total_duration = 60 * len(video_files)  # Assume 1 minute per file
            logger.warning(f"Could not determine video durations. Using fallback: {total_duration:.2f}s")
        
        # Calculate estimated number of B-roll clips needed
        edge_buffer = 5.0
        usable_duration = max(0, total_duration - (2 * edge_buffer))
        estimated_broll_count = max(1, int(usable_duration / fixed_interval))
        
        # Use the minimum of the estimated count and the user-specified num_broll
        broll_limit = min(estimated_broll_count, num_broll) if num_broll > 0 else estimated_broll_count
        
        logger.info(f"Total video duration: {total_duration:.2f}s, estimated B-roll clips needed: {estimated_broll_count}")
        logger.info(f"Using B-roll limit: {broll_limit}")
        
        # Step 2: Search and download B-roll for these keywords
        broll_paths = await self.broll_service.get_broll_for_keywords(
            keywords=keywords,
            output_dir=output_dir,
            orientation=orientation,
            size=video_size,
            max_videos=broll_limit,  # Use our calculated limit
            min_duration=broll_duration,
            max_duration=broll_duration * 2
        )
        
        if not broll_paths:
            logger.warning("No B-roll videos found, returning original videos merged")
            # Just merge the videos without B-roll
            return await self.video_processor._merge_videos(video_files, output_path)
            
        # Step 3: Determine insertion points across video files
        insertion_points = self.content_analyzer.determine_broll_points(
            video_files=video_files,
            num_points=len(broll_paths),  # Limit to the number of B-roll videos we have
            edge_buffer=5.0,
            fixed_interval=fixed_interval  # Use our fixed interval
        )
        
        # Step 4: Insert B-roll at determined points
        processed_path = await self.video_processor.insert_broll_multifile(
            video_files=video_files,
            broll_paths=broll_paths,
            insertion_points=insertion_points,
            output_path=output_path,
            broll_duration=broll_duration
        )
        
        logger.info(f"B-roll post-processing complete: {processed_path}")
        return processed_path

# Standalone function for ease of import
async def apply_broll_post_processing(
    video_path_or_files: Union[str, List[str]],
    text: str,
    output_dir: Path,
    output_filename: Optional[str] = None,
    num_broll: int = 2,
    broll_duration: float = 5.0,
    api_key: Optional[str] = None,
    fixed_interval: float = 7.0  # New parameter
) -> str:
    """
    Apply B-roll post-processing to video(s).
    
    Args:
        video_path_or_files: Path to video or list of video paths
        text: Original text content
        output_dir: Directory for output and intermediate files
        output_filename: Output filename (defaults to "final_with_broll.mp4")
        num_broll: Maximum number of B-roll segments to insert
        broll_duration: Duration of each B-roll segment in seconds
        api_key: Pexels API key (defaults to PEXELS_API_KEY environment variable)
        fixed_interval: Interval in seconds between B-roll insertions
        
    Returns:
        Path to the processed video
    """
    pipeline = PostProcessingPipeline(api_key)
    
    if output_filename is None:
        output_filename = "final_with_broll.mp4"
        
    output_path = output_dir / output_filename
    
    # Process single or multiple files
    if isinstance(video_path_or_files, str):
        return await pipeline.process(
            text=text,
            video_path=video_path_or_files,
            output_path=str(output_path),
            output_dir=output_dir,
            num_broll=num_broll,
            broll_duration=broll_duration,
            fixed_interval=fixed_interval
        )
    else:
        return await pipeline.process_multifile(
            text=text,
            video_files=video_path_or_files,
            output_path=str(output_path),
            output_dir=output_dir,
            num_broll=num_broll,
            broll_duration=broll_duration,
            fixed_interval=fixed_interval
        )