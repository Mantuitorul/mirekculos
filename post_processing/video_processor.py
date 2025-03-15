#!/usr/bin/env python3
# post_processing/video_processor.py
"""
Video processing for B-roll insertion.
Handles cutting, splicing, and maintaining audio continuity.
"""

import os
import logging
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Direct imports from specific modules instead of using moviepy.editor
import moviepy
print(moviepy.__version__)
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import concatenate_videoclips

# Configure logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Processes videos for B-roll insertion"""
    
    def __init__(self):
        """Initialize the video processor"""
        pass
        
    async def insert_broll(
        self,
        video_path: str,
        broll_paths: List[str],
        insertion_points: List[Dict[str, Any]],
        output_path: str,
        broll_duration: float = 5.0,
        transition_duration: float = 0.5,
        output_width: int = 1280,
        output_height: int = 720
    ) -> str:
        """
        Insert B-roll footage into a video at specified points.
        
        Args:
            video_path: Path to the main video
            broll_paths: Paths to B-roll footage
            insertion_points: Points to insert B-roll
            output_path: Path to save the output video
            broll_duration: Duration of each B-roll insert
            transition_duration: Duration of fade transition
            output_width: Width of output video
            output_height: Height of output video
            
        Returns:
            Path to the output video
        """
        logger.info(f"Inserting B-roll into video: {video_path}")
        logger.info(f"B-roll footage: {broll_paths}")
        logger.info(f"Insertion points: {insertion_points}")
        
        # Use asyncio run_in_executor to run CPU-bound moviepy operations in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._insert_broll_sync,
            video_path, broll_paths, insertion_points, output_path, 
            broll_duration, transition_duration, output_width, output_height
        )
    
    def _insert_broll_sync(
        self,
        video_path: str,
        broll_paths: List[str],
        insertion_points: List[Dict[str, Any]],
        output_path: str,
        broll_duration: float = 5.0,
        transition_duration: float = 0.5,
        output_width: int = 1280,
        output_height: int = 720
    ) -> str:
        """
        Synchronous implementation of B-roll insertion.
        """
        # Load the main video
        main_clip = VideoFileClip(video_path)
        
        # Load B-roll clips
        broll_clips = []
        for path in broll_paths:
            try:
                clip = VideoFileClip(path)
                # Resize B-roll to match main video dimensions
                if clip.w != output_width or clip.h != output_height:
                    clip = clip.resize(newsize=(output_width, output_height))
                broll_clips.append(clip)
            except Exception as e:
                logger.error(f"Error loading B-roll clip {path}: {str(e)}")
        
        if not broll_clips:
            logger.warning("No B-roll clips loaded, returning original video")
            main_clip.close()
            return video_path
            
        # Sort insertion points by time
        insertion_points = sorted(insertion_points, key=lambda x: x.get("global_time", 0))
        
        # Create segment list for final composition
        segments = []
        last_end_time = 0
        
        try:
            # Process each insertion point
            for i, point in enumerate(insertion_points):
                point_time = point.get("global_time", 0)
                
                # Add main video segment before this B-roll
                if point_time > last_end_time:
                    segment = main_clip.subclip(last_end_time, point_time)
                    segments.append(segment)
                    
                # Select and insert B-roll for this point
                # Try to use different B-roll for each point
                broll_idx = i % len(broll_clips) if i < len(broll_clips) else random.randint(0, len(broll_clips) - 1)
                broll = broll_clips[broll_idx]
                
                # Calculate actual B-roll duration (min of requested duration and B-roll length)
                actual_duration = min(broll_duration, broll.duration)
                
                # Extract audio from main clip for this segment
                main_audio = main_clip.subclip(point_time, point_time + actual_duration).audio
                
                # Create B-roll segment with main audio
                broll_segment = broll.subclip(0, actual_duration).set_audio(main_audio)
                segments.append(broll_segment)
                
                # Update last end time
                last_end_time = point_time + actual_duration
            
            # Add final segment if needed
            if last_end_time < main_clip.duration:
                final_segment = main_clip.subclip(last_end_time)
                segments.append(final_segment)
                
            # Create final composition
            logger.info(f"Creating final composition with {len(segments)} segments")
            final_clip = concatenate_videoclips(segments)
            
            # Write output file
            logger.info(f"Writing output to {output_path}")
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                logger=None
            )
            
            logger.info(f"B-roll insertion complete: {output_path}")
            return output_path
            
        finally:
            # Clean up
            main_clip.close()
            for clip in broll_clips:
                clip.close()
        
    async def insert_broll_multifile(
        self,
        video_files: List[str],
        broll_paths: List[str],
        insertion_points: List[Dict[str, Any]],
        output_path: str,
        broll_duration: float = 5.0,
        transition_duration: float = 0.5,
        output_width: int = 1280,
        output_height: int = 720
    ) -> str:
        """
        Insert B-roll footage into multiple video files at specified points.
        
        Args:
            video_files: Paths to the original video segments
            broll_paths: Paths to B-roll footage
            insertion_points: Points to insert B-roll (must include file_index)
            output_path: Path to save the output video
            broll_duration: Duration of each B-roll insert
            transition_duration: Duration of fade transition
            output_width: Width of output video
            output_height: Height of output video
            
        Returns:
            Path to the output video
        """
        logger.info(f"Inserting B-roll into {len(video_files)} video segments")
        
        # Use asyncio run_in_executor to run CPU-bound moviepy operations in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._insert_broll_multifile_sync,
            video_files, broll_paths, insertion_points, output_path, 
            broll_duration, transition_duration, output_width, output_height
        )
    
    def _insert_broll_multifile_sync(
        self,
        video_files: List[str],
        broll_paths: List[str],
        insertion_points: List[Dict[str, Any]],
        output_path: str,
        broll_duration: float = 5.0,
        transition_duration: float = 0.5,
        output_width: int = 1280,
        output_height: int = 720
    ) -> str:
        """
        Synchronous implementation of B-roll insertion for multiple files.
        """
        # Group insertion points by file index
        points_by_file = {}
        for point in insertion_points:
            file_idx = point.get("file_index", 0)
            file_time = point.get("file_time", 0)
            
            if file_idx not in points_by_file:
                points_by_file[file_idx] = []
                
            points_by_file[file_idx].append({
                "time": file_time,
                "global_time": point.get("global_time", 0)
            })
        
        # Load B-roll clips
        broll_clips = []
        for path in broll_paths:
            try:
                clip = VideoFileClip(path)
                # Resize B-roll to match output dimensions
                if clip.w != output_width or clip.h != output_height:
                    clip = clip.resize((output_width, output_height))
                broll_clips.append(clip)
            except Exception as e:
                logger.error(f"Error loading B-roll clip {path}: {str(e)}")
        
        if not broll_clips:
            logger.warning("No B-roll clips loaded, returning original videos")
            return self._merge_videos_sync(video_files, output_path)
        
        # Process each video file
        processed_segments = []
        
        try:
            for file_idx, file_path in enumerate(video_files):
                # If this file has insertion points
                if file_idx in points_by_file:
                    # Load video
                    try:
                        clip = VideoFileClip(file_path)
                        
                        # Sort points by time
                        file_points = sorted(points_by_file[file_idx], key=lambda x: x["time"])
                        
                        # Create segments for this file
                        file_segments = []
                        last_end_time = 0
                        
                        # Process each insertion point
                        for i, point in enumerate(file_points):
                            point_time = point["time"]
                            
                            # Add segment before B-roll
                            if point_time > last_end_time:
                                segment = clip.subclip(last_end_time, point_time)
                                file_segments.append(segment)
                                
                            # Select B-roll
                            broll_idx = i % len(broll_clips)
                            broll = broll_clips[broll_idx]
                            
                            # Calculate duration
                            actual_duration = min(broll_duration, broll.duration)
                            
                            # Get audio from main clip
                            if point_time + actual_duration <= clip.duration:
                                main_audio = clip.subclip(point_time, point_time + actual_duration).audio
                                
                                # Create B-roll segment with main audio
                                broll_segment = broll.subclip(0, actual_duration).set_audio(main_audio)
                                file_segments.append(broll_segment)
                                
                                # Update last end time
                                last_end_time = point_time + actual_duration
                            else:
                                # If B-roll would extend past clip end, don't add it
                                logger.warning(f"B-roll at {point_time:.2f}s extends past end of clip, skipping")
                        
                        # Add final segment
                        if last_end_time < clip.duration:
                            final_segment = clip.subclip(last_end_time)
                            file_segments.append(final_segment)
                            
                        # Concatenate file segments
                        if file_segments:
                            processed_file = concatenate_videoclips(file_segments)
                            processed_segments.append(processed_file)
                        
                        # Close clip
                        clip.close()
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        # Use original clip if processing fails
                        processed_segments.append(VideoFileClip(file_path))
                else:
                    # No insertion points for this file, use as is
                    processed_segments.append(VideoFileClip(file_path))
            
            # Concatenate all processed segments
            logger.info(f"Creating final composition with {len(processed_segments)} segments")
            final_clip = concatenate_videoclips(processed_segments)
            
            # Write output file
            logger.info(f"Writing output to {output_path}")
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                logger=None
            )
                
            logger.info(f"B-roll insertion complete: {output_path}")
            return output_path
            
        finally:
            # Clean up
            for clip in processed_segments + broll_clips:
                try:
                    clip.close()
                except:
                    pass
        
    async def merge_videos(self, video_files: List[str], output_path: str) -> str:
        """
        Async wrapper for merging videos without B-roll.
        
        Args:
            video_files: List of video files to merge
            output_path: Output path
            
        Returns:
            Path to merged video
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._merge_videos_sync, video_files, output_path)
    
    def _merge_videos_sync(self, video_files: List[str], output_path: str) -> str:
        """
        Synchronous implementation of video merging without B-roll.
        
        Args:
            video_files: List of video files to merge
            output_path: Output path
            
        Returns:
            Path to merged video
        """
        clips = [VideoFileClip(f) for f in video_files]
        try:
            merged = concatenate_videoclips(clips)
            merged.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                logger=None
            )
            return output_path
        finally:
            for clip in clips:
                clip.close()