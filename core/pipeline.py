#!/usr/bin/env python3
"""
Main Pipeline class for orchestrating the video generation process.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .config import Config, ensure_output_dir

class Pipeline:
    """
    Main pipeline for text-to-video generation.
    
    This class orchestrates the entire process:
    1. Structure input text into video segments using ChatGPT
    2. Generate videos for each segment with HeyGen
    3. Process B-roll segments with Pexels footage
    4. Merge all segments into final video
    """
    
    def __init__(
        self, 
        config: Optional[Config] = None,
        width: int = 720,
        height: int = 1280,
        output_dir: Union[str, Path] = "output",
        debug_mode: bool = False,
        debug_dir: Optional[Union[str, Path]] = "debug_output"
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object (created automatically if None)
            width: Video width in pixels
            height: Video height in pixels
            output_dir: Directory for output files
            debug_mode: Whether to enable debug mode
            debug_dir: Directory for debug files (if debug_mode is True)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config if config else Config()
        self.width = width
        self.height = height
        self.output_dir = ensure_output_dir(Path(output_dir))
        self.segments_dir = ensure_output_dir(self.output_dir / "segments")
        self.debug_mode = debug_mode
        self.debug_dir = Path(debug_dir) if debug_dir else None
        
        if self.debug_mode and self.debug_dir:
            ensure_output_dir(self.debug_dir)
    
    async def run(
        self,
        text: str,
        front_avatar_id: str,
        side_avatar_id: str,
        heygen_voice_id: Optional[str] = None,
        heygen_emotion: Optional[str] = None,
        avatar_style: str = "normal",
        background_color: str = "#008000",
        output_filename: str = "final_output.mp4"
    ) -> Dict[str, Any]:
        """
        Run the full pipeline.
        
        Args:
            text: Input text for video generation
            front_avatar_id: HeyGen avatar ID for front shots
            side_avatar_id: HeyGen avatar ID for side shots
            heygen_voice_id: HeyGen voice ID (optional)
            heygen_emotion: HeyGen voice emotion (optional)
            avatar_style: Avatar style (normal, happy, etc.)
            background_color: Background color (hex)
            output_filename: Filename for the final output video
            
        Returns:
            Dict with pipeline results
        """
        self.logger.info("Starting video generation pipeline")
        
        try:
            # Step 1: Structure the text with ChatGPT
            segments = await self._structure_text(text)
            
            # Step 2: Generate videos with HeyGen
            processed_segments = await self._generate_videos(
                segments=segments,
                front_avatar_id=front_avatar_id,
                side_avatar_id=side_avatar_id,
                voice_id=heygen_voice_id,
                emotion=heygen_emotion,
                avatar_style=avatar_style,
                background_color=background_color
            )
            
            # Step 3: Extract audio from B-roll segments
            if any(s.get("is_broll", False) for s in processed_segments):
                await self._extract_broll_audio(processed_segments)
                
                # Step 4: Generate Pexels B-roll videos
                await self._generate_pexels_broll(processed_segments)
            
            # Step 5: Merge final video
            final_video_path = await self._merge_final_video(
                processed_segments, 
                output_filename
            )
            
            # Count each segment type
            segment_counts = {
                "front": len([s for s in segments if s["segment_shot"].lower() == "front"]),
                "side": len([s for s in segments if s["segment_shot"].lower() == "side"]),
                "broll": len([s for s in segments if s["segment_shot"].lower() == "broll"])
            }
            
            # Create result object
            result = {
                "success": True,
                "final_video": final_video_path,
                "total_segments": len(segments),
                "segment_counts": segment_counts,
                "broll_segments": [s for s in processed_segments if s.get("is_broll", False)],
                "has_broll": segment_counts["broll"] > 0
            }
            
            self.logger.info(f"Pipeline completed! Final video: {final_video_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _structure_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Structure input text into video segments using ChatGPT.
        
        Args:
            text: Input text
            
        Returns:
            List of structured segments
        """
        from text.processing import VideoStructurer
        
        self.logger.info("Using ChatGPT to structure video with front, side, and broll segments")
        structurer = VideoStructurer(
            api_key=self.config.openai_api_key,
            debug_mode=self.debug_mode,
            debug_dir=self.debug_dir
        )
        segments = structurer.structure_article(text)
        
        # Save segments to file
        segments_file = self.output_dir / "segments.json"
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created {len(segments)} segments with ChatGPT")
        return segments
    
    async def _generate_videos(
        self,
        segments: List[Dict[str, Any]],
        front_avatar_id: str,
        side_avatar_id: str,
        voice_id: Optional[str] = None,
        emotion: Optional[str] = None,
        avatar_style: str = "normal",
        background_color: str = "#008000"
    ) -> List[Dict[str, Any]]:
        """
        Generate videos for all segments using HeyGen.
        
        Args:
            segments: Structured segments
            front_avatar_id: HeyGen avatar ID for front shots
            side_avatar_id: HeyGen avatar ID for side shots
            voice_id: HeyGen voice ID (optional)
            emotion: HeyGen voice emotion (optional)
            avatar_style: Avatar style (normal, happy, etc.)
            background_color: Background color (hex)
            
        Returns:
            List of processed segment information
        """
        from video.heygen import create_heygen_video, poll_video_status, download_video
        
        self.logger.info(f"Processing {len(segments)} segments with HeyGen")
        
        # Get HeyGen API keys
        api_keys = self.config.heygen_api_keys
        if not api_keys:
            raise ValueError("No HeyGen API keys available")
        
        # Process segments in parallel if multiple API keys are available
        if len(api_keys) > 1:
            return await self._generate_videos_parallel(
                segments=segments,
                api_keys=api_keys,
                front_avatar_id=front_avatar_id,
                side_avatar_id=side_avatar_id,
                voice_id=voice_id,
                emotion=emotion,
                avatar_style=avatar_style,
                background_color=background_color
            )
        
        # Otherwise, process segments sequentially
        results = []
        api_key = api_keys[0]
        
        for i, segment in enumerate(segments):
            segment_text = segment["segment_text"]
            segment_shot = segment["segment_shot"]
            
            # Determine avatar ID based on shot type
            if segment_shot.lower() == "front":
                current_avatar_id = front_avatar_id
                is_broll = False
            elif segment_shot.lower() == "side":
                current_avatar_id = side_avatar_id
                is_broll = False
            elif segment_shot.lower() == "broll":
                # Important: Process broll as "front" to get proper audio
                current_avatar_id = front_avatar_id
                is_broll = True
                self.logger.info(f"Processing broll segment {i+1}/{len(segments)} as front avatar for audio generation")
            else:
                # Default to front for unknown types
                self.logger.warning(f"Unknown segment type: {segment_shot}, defaulting to front")
                current_avatar_id = front_avatar_id
                is_broll = False
            
            # Create video
            self.logger.info(f"Creating video for segment {i+1}/{len(segments)}: {segment_shot}")
            video_id = await create_heygen_video(
                avatar_id=current_avatar_id,
                avatar_style=avatar_style,
                background_color=background_color,
                width=self.width,
                height=self.height,
                api_key=api_key,
                input_text=segment_text,
                voice_id=voice_id,
                emotion=emotion
            )
            
            # Poll status
            self.logger.info(f"Polling status for video {i+1}: {video_id}")
            status = await poll_video_status(video_id, api_key)
            video_url = status.get("video_url")
            if not video_url:
                raise ValueError(f"No video URL in status for video {video_id}")
            
            # Download video
            output_path = self.segments_dir / f"segment_{i}.mp4"
            path = await download_video(video_url, str(output_path))
            
            results.append({
                "path": path,
                "order": i,
                "shot_type": segment_shot,
                "is_broll": is_broll,
                "segment_text": segment_text,
                "instructions": segment.get("instructions", "") if is_broll else ""
            })
            
            self.logger.info(f"Downloaded segment {i+1}: {path}")
        
        return results
    
    async def _generate_videos_parallel(
        self,
        segments: List[Dict[str, Any]],
        api_keys: List[str],
        front_avatar_id: str,
        side_avatar_id: str,
        voice_id: Optional[str] = None,
        emotion: Optional[str] = None,
        avatar_style: str = "normal",
        background_color: str = "#008000"
    ) -> List[Dict[str, Any]]:
        """
        Generate videos for all segments in parallel using multiple API keys.
        
        Args:
            segments: Structured segments
            api_keys: List of HeyGen API keys
            front_avatar_id: HeyGen avatar ID for front shots
            side_avatar_id: HeyGen avatar ID for side shots
            voice_id: HeyGen voice ID (optional)
            emotion: HeyGen voice emotion (optional)
            avatar_style: Avatar style (normal, happy, etc.)
            background_color: Background color (hex)
            
        Returns:
            List of processed segment information
        """
        from video.heygen import create_heygen_video, poll_video_status, download_video
        
        self.logger.info(f"Processing {len(segments)} segments in parallel using {len(api_keys)} API keys")
        
        # Create tasks list and results list
        tasks = []
        results = []
        available_keys = asyncio.Queue()
        
        # Initialize the queue with available API keys
        for key in api_keys:
            await available_keys.put(key)
        
        async def process_segment(segment_index, segment):
            # Get an available API key
            api_key = await available_keys.get()
            try:
                self.logger.info(f"Processing segment {segment_index+1}/{len(segments)} with API key: {api_key[:8]}...")
                
                segment_text = segment["segment_text"]
                segment_shot = segment["segment_shot"]
                
                # Determine avatar ID based on shot type
                if segment_shot.lower() == "front":
                    current_avatar_id = front_avatar_id
                    is_broll = False
                elif segment_shot.lower() == "side":
                    current_avatar_id = side_avatar_id
                    is_broll = False
                elif segment_shot.lower() == "broll":
                    # Important: Process broll as "front" to get proper audio
                    current_avatar_id = front_avatar_id
                    is_broll = True
                    self.logger.info(f"Processing broll segment {segment_index+1} as front avatar for audio generation")
                else:
                    # Default to front for unknown types
                    self.logger.warning(f"Unknown segment type: {segment_shot}, defaulting to front")
                    current_avatar_id = front_avatar_id
                    is_broll = False
                
                # Create video
                self.logger.info(f"Creating video for segment {segment_index+1}/{len(segments)}: {segment_shot}")
                video_id = await create_heygen_video(
                    avatar_id=current_avatar_id,
                    avatar_style=avatar_style,
                    background_color=background_color,
                    width=self.width,
                    height=self.height,
                    api_key=api_key,
                    input_text=segment_text,
                    voice_id=voice_id,
                    emotion=emotion
                )
                
                # Poll status
                self.logger.info(f"Polling status for video {segment_index+1}: {video_id}")
                status = await poll_video_status(video_id, api_key)
                video_url = status.get("video_url")
                if not video_url:
                    raise ValueError(f"No video URL in status for video {video_id}")
                
                # Download video
                output_path = self.segments_dir / f"segment_{segment_index}.mp4"
                path = await download_video(video_url, str(output_path))
                
                return {
                    "path": path,
                    "order": segment_index,
                    "shot_type": segment_shot,
                    "is_broll": is_broll,
                    "segment_text": segment_text,
                    "instructions": segment.get("instructions", "") if is_broll else ""
                }
            finally:
                # Always put the API key back in the queue
                await available_keys.put(api_key)
        
        # Create a task for each segment
        for i, segment in enumerate(segments):
            task = asyncio.create_task(process_segment(i, segment))
            tasks.append(task)
        
        # Wait for all tasks to complete
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task failed: {str(e)}")
                # Let other tasks continue
        
        # Sort results by order
        results.sort(key=lambda x: x["order"])
        
        return results
    
    async def _extract_broll_audio(self, segments_info: List[Dict[str, Any]]) -> None:
        """
        Extract audio from B-roll segments.
        
        Args:
            segments_info: Processed segment information
        """
        from audio.processing import extract_audio_from_segments
        
        self.logger.info("Extracting audio from B-roll segments")
        broll_segments = [s for s in segments_info if s.get("is_broll", False)]
        
        if not broll_segments:
            self.logger.info("No B-roll segments to process")
            return
        
        extract_audio_from_segments(broll_segments, self.output_dir)
        
        self.logger.info(f"Extracted audio from {len(broll_segments)} B-roll segments")
    
    async def _generate_pexels_broll(self, segments_info: List[Dict[str, Any]]) -> None:
        """
        Generate Pexels B-roll videos for B-roll segments.
        
        Args:
            segments_info: Processed segment information
        """
        from video.broll import BRollService, create_broll_segments
        from text.processing import ContentAnalyzer
        
        self.logger.info("Creating Pexels B-roll videos")
        broll_segments = [s for s in segments_info if s.get("is_broll", False)]
        
        if not broll_segments:
            self.logger.info("No B-roll segments to process")
            return
        
        pexels_api_key = self.config.pexels_api_key
        if not pexels_api_key:
            self.logger.warning("Pexels API key is not available. Using original segments for B-roll.")
            return
        
        # Create content analyzer for keyword extraction
        content_analyzer = ContentAnalyzer()
        
        # Create B-roll service
        broll_service = BRollService(pexels_api_key)
        
        # Create B-roll segments
        await create_broll_segments(
            segments=broll_segments,
            keywords_extractor=content_analyzer.extract_keywords,
            broll_service=broll_service,
            output_dir=self.output_dir
        )
        
        self.logger.info(f"Created B-roll videos for {len(broll_segments)} segments")
    
    async def _merge_final_video(
        self, 
        segments_info: List[Dict[str, Any]], 
        output_filename: str = "final_output.mp4"
    ) -> str:
        """
        Merge all segments into the final video.
        
        Args:
            segments_info: Processed segment information
            output_filename: Filename for the final output video
            
        Returns:
            Path to the final video
        """
        from video.merger import merge_with_broll
        
        self.logger.info("Merging final video")
        output_path = self.output_dir / output_filename
        
        # Check if any segments have B-roll
        has_broll = any(s.get("is_broll", False) and "broll_video" in s for s in segments_info)
        
        if has_broll:
            # Use B-roll merge
            result = await merge_with_broll(
                segments=segments_info,
                output_path=str(output_path),
                width=self.width,
                height=self.height
            )
        else:
            # Use simple merge
            from video.merger import merge_videos
            segment_paths = [s["path"] for s in segments_info]
            result = await merge_videos(segment_paths, str(output_path))
        
        self.logger.info(f"Merged final video: {result}")
        return result