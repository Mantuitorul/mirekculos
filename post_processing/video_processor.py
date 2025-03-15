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

import moviepy
print(moviepy.__version__)
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import concatenate_videoclips
# Removed: from moviepy.video.fx.all import resize

# Configure logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Processes videos for B-roll insertion"""
    
    def __init__(self):
        """Initialize the video processor"""
        pass
        
    def generate_regular_insertion_points(self, video_path: str, interval: float = 7.0) -> List[Dict[str, Any]]:
        """
        Generate insertion points at regular intervals throughout the video.
        
        Args:
            video_path: Path to the video file
            interval: Time interval between B-roll insertions in seconds
            
        Returns:
            List of insertion points
        """
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        
        # Start at the first interval (e.g., 7 seconds in)
        # This skips inserting B-roll at the very beginning
        insertion_points = []
        current_time = interval
        
        while current_time < duration - interval:  # Avoid inserting too close to the end
            insertion_points.append({
                "global_time": current_time,
                "file_index": 0,
                "file_path": video_path,
                "file_time": current_time
            })
            current_time += interval
            
        logger.info(f"Generated {len(insertion_points)} insertion points at {interval}s intervals")
        return insertion_points
        
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
        logger.info(f"Inserting B-roll into video: {video_path}")
        logger.info(f"B-roll footage: {broll_paths}")
        logger.info(f"Insertion points: {insertion_points}")
        
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
        main_clip = VideoFileClip(video_path)
        
        broll_clips = []
        for path in broll_paths:
            try:
                clip = VideoFileClip(path)
                # Resize B-roll to match main video dimensions using the new API
                if clip.w != output_width or clip.h != output_height:
                    clip = clip.resized(new_size=(output_width, output_height))
                broll_clips.append(clip)
            except Exception as e:
                logger.error(f"Error loading B-roll clip {path}: {str(e)}")
        
        if not broll_clips:
            logger.warning("No B-roll clips loaded, returning original video")
            main_clip.close()
            return video_path
            
        insertion_points = sorted(insertion_points, key=lambda x: x.get("global_time", 0))
        segments = []
        last_end_time = 0
        
        try:
            for i, point in enumerate(insertion_points):
                point_time = point.get("global_time", 0)
                if point_time > last_end_time:
                    segment = main_clip.subclipped(last_end_time, point_time)
                    segments.append(segment)
                    
                broll_idx = i % len(broll_clips) if i < len(broll_clips) else random.randint(0, len(broll_clips) - 1)
                broll = broll_clips[broll_idx]
                actual_duration = min(broll_duration, broll.duration)
                main_audio = main_clip.subclipped(point_time, point_time + actual_duration).audio
                broll_segment = broll.subclipped(0, actual_duration).with_audio(main_audio)
                segments.append(broll_segment)
                last_end_time = point_time + actual_duration
            
            if last_end_time < main_clip.duration:
                final_segment = main_clip.subclipped(last_end_time)
                segments.append(final_segment)
                
            logger.info(f"Creating final composition with {len(segments)} segments")
            final_clip = concatenate_videoclips(segments)
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
        logger.info(f"Inserting B-roll into {len(video_files)} video segments")
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
        
        broll_clips = []
        for path in broll_paths:
            try:
                clip = VideoFileClip(path)
                if clip.w != output_width or clip.h != output_height:
                    clip = clip.resized(new_size=(output_width, output_height))
                broll_clips.append(clip)
            except Exception as e:
                logger.error(f"Error loading B-roll clip {path}: {str(e)}")
        
        if not broll_clips:
            logger.warning("No B-roll clips loaded, returning original videos")
            return self._merge_videos_sync(video_files, output_path)
        
        processed_segments = []
        
        try:
            for file_idx, file_path in enumerate(video_files):
                if file_idx in points_by_file:
                    try:
                        clip = VideoFileClip(file_path)
                        file_points = sorted(points_by_file[file_idx], key=lambda x: x["time"])
                        file_segments = []
                        last_end_time = 0
                        for i, point in enumerate(file_points):
                            point_time = point["time"]
                            if point_time > last_end_time:
                                segment = clip.subclipped(last_end_time, point_time)
                                file_segments.append(segment)
                            broll_idx = i % len(broll_clips)
                            broll = broll_clips[broll_idx]
                            actual_duration = min(broll_duration, broll.duration)
                            if point_time + actual_duration <= clip.duration:
                                main_audio = clip.subclipped(point_time, point_time + actual_duration).audio
                                broll_segment = broll.subclipped(0, actual_duration).with_audio(main_audio)
                                file_segments.append(broll_segment)
                                last_end_time = point_time + actual_duration
                            else:
                                logger.warning(f"B-roll at {point_time:.2f}s extends past end of clip, skipping")
                        
                        if last_end_time < clip.duration:
                            final_segment = clip.subclipped(last_end_time)
                            file_segments.append(final_segment)
                            
                        if file_segments:
                            processed_file = concatenate_videoclips(file_segments)
                            processed_segments.append(processed_file)
                        clip.close()
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        processed_segments.append(VideoFileClip(file_path))
                else:
                    processed_segments.append(VideoFileClip(file_path))
            
            logger.info(f"Creating final composition with {len(processed_segments)} segments")
            final_clip = concatenate_videoclips(processed_segments)
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
            for clip in processed_segments + broll_clips:
                try:
                    clip.close()
                except:
                    pass
        
    async def merge_videos(self, video_files: List[str], output_path: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._merge_videos_sync, video_files, output_path)
    
    def _merge_videos_sync(self, video_files: List[str], output_path: str) -> str:
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