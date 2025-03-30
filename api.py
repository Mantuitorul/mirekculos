 #!/usr/bin/env python3
"""
FastAPI endpoint for the video generation pipeline.
Provides an HTTP API for generating reels with specified parameters.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from core import Pipeline, Config
from audio.scilence_remover import remove_silence

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Mirekculos AI Video Generator",
    description="API for generating AI videos using HeyGen avatars and Pexels B-roll footage",
    version="0.1.0"
)

# Store current generation jobs 
active_processes = {}

class VideoRequest(BaseModel):
    """Request model for video generation."""
    text: str = Field(..., description="Input text for video generation")
    voice_id_1: str = Field(..., description="HeyGen voice ID for the video")
    look_id_1: str = Field(..., description="HeyGen avatar ID for front shots")
    look_id_2: str = Field(..., description="HeyGen avatar ID for side shots")
    remove_silence: bool = Field(True, description="Whether to remove silence from the final video")
    silence_threshold: float = Field(-30, description="Silence threshold in dB")
    min_silence_duration: float = Field(0.3, description="Minimum silence duration in seconds")
    keep_silence_ratio: float = Field(0.2, description="Portion of silence to keep (0-1)")
    emotion: Optional[str] = Field(None, description="HeyGen voice emotion (Excited, Friendly, Serious, Soothing, Broadcaster)")
    background_color: str = Field("#008000", description="Background color (hex)")
    width: int = Field(720, description="Video width in pixels")
    height: int = Field(1280, description="Video height in pixels")
    debug: bool = Field(False, description="Enable debug mode")

class ProcessStatus(BaseModel):
    """Status model for video generation processes."""
    process_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

async def generate_video_task(
    process_id: str,
    params: VideoRequest
) -> None:
    """
    Background task to generate a video with the specified parameters.
    
    Args:
        process_id: Unique identifier for this process
        params: VideoRequest with parameters for video generation
    """
    try:
        # Update process status to "processing"
        active_processes[process_id]["status"] = "processing"
        
        # Create output directory for this process
        output_dir = Path("output") / process_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create pipeline instance
        pipeline = Pipeline(
            width=params.width,
            height=params.height,
            output_dir=str(output_dir),
            debug_mode=params.debug,
            debug_dir=str(output_dir / "debug") if params.debug else None
        )
        
        # Run the pipeline
        output_filename = f"final_output_{process_id}.mp4"
        result = await pipeline.run(
            text=params.text,
            front_avatar_id=params.look_id_1,
            side_avatar_id=params.look_id_2,
            heygen_voice_id=params.voice_id_1,
            heygen_emotion=params.emotion,
            background_color=params.background_color,
            output_filename=output_filename
        )
        
        # Process result
        if result["success"]:
            final_video_path = result["final_video"]
            
            # Apply silence removal if requested
            if params.remove_silence:
                logger.info(f"Removing silence from video: {final_video_path}")
                no_silence_path = Path(final_video_path).parent / f"{Path(final_video_path).stem}_no_silence{Path(final_video_path).suffix}"
                
                success = remove_silence(
                    final_video_path,
                    no_silence_path,
                    silence_threshold=params.silence_threshold,
                    min_silence_duration=params.min_silence_duration,
                    keep_ratio=params.keep_silence_ratio
                )
                
                if success:
                    logger.info(f"Successfully removed silence: {no_silence_path}")
                    result["final_video_no_silence"] = str(no_silence_path)
                    # Update the final video to be the no-silence version
                    result["original_video"] = result["final_video"]
                    result["final_video"] = str(no_silence_path)
                else:
                    logger.error("Failed to remove silence")
            
            # Update process status
            active_processes[process_id]["status"] = "completed"
            active_processes[process_id]["result"] = result
        else:
            # Handle failure
            active_processes[process_id]["status"] = "failed"
            active_processes[process_id]["error"] = result.get("error", "Unknown error")
        
    except Exception as e:
        logger.exception(f"Error in process {process_id}: {str(e)}")
        active_processes[process_id]["status"] = "failed"
        active_processes[process_id]["error"] = str(e)
    finally:
        active_processes[process_id]["completed_at"] = datetime.now().isoformat()

@app.post("/generate-reel", response_model=ProcessStatus)
async def generate_reel(params: VideoRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Start a new video generation process.
    
    Args:
        params: Parameters for video generation
    
    Returns:
        Process ID and initial status
    """
    # Generate a unique process ID
    process_id = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create process entry
    active_processes[process_id] = {
        "process_id": process_id,
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None
    }
    
    # Start background task for video generation
    background_tasks.add_task(generate_video_task, process_id, params)
    
    return JSONResponse(
        content=active_processes[process_id],
        status_code=202
    )

@app.get("/process/{process_id}", response_model=ProcessStatus)
async def get_process_status(process_id: str) -> JSONResponse:
    """
    Get the status of a process.
    
    Args:
        process_id: Process ID to query
    
    Returns:
        Current process status
    """
    if process_id not in active_processes:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")
    
    return JSONResponse(content=active_processes[process_id])

@app.get("/processes")
async def list_processes() -> JSONResponse:
    """
    List all processes and their statuses.
    
    Returns:
        List of all processes
    """
    return JSONResponse(content=list(active_processes.values()))

@app.get("/download/{process_id}")
async def download_video(process_id: str, no_silence: bool = True) -> FileResponse:
    """
    Download the generated video.
    
    Args:
        process_id: Process ID of the video to download
        no_silence: Whether to download the version with silence removed (if available)
    
    Returns:
        Video file
    """
    if process_id not in active_processes:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")
    
    process = active_processes[process_id]
    
    if process["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Process {process_id} is not completed (current status: {process['status']})"
        )
    
    if not process["result"]:
        raise HTTPException(status_code=500, detail=f"No result available for process {process_id}")
    
    # Choose the right video file based on no_silence parameter
    if no_silence and "final_video_no_silence" in process["result"]:
        video_path = process["result"]["final_video_no_silence"]
    else:
        video_path = process["result"]["final_video"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    return FileResponse(
        path=video_path, 
        media_type="video/mp4",
        filename=f"reel_{process_id}.mp4"
    )

@app.delete("/process/{process_id}")
async def delete_process(process_id: str) -> JSONResponse:
    """
    Delete a process and its associated files.
    
    Args:
        process_id: Process ID to delete
    
    Returns:
        Confirmation message
    """
    if process_id not in active_processes:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")
    
    # Get output directory for this process
    output_dir = Path("output") / process_id
    
    # Delete files if directory exists
    if output_dir.exists():
        import shutil
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            logger.error(f"Error deleting process directory: {str(e)}")
    
    # Remove from active processes
    process = active_processes.pop(process_id)
    
    return JSONResponse(
        content={"message": f"Process {process_id} deleted successfully"}
    )

@app.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint.
    
    Returns:
        API status
    """
    # Check for necessary API keys
    config = Config()
    api_keys_available = {
        "heygen": len(config.heygen_api_keys) > 0,
        "openai": config.openai_api_key is not None,
        "pexels": config.pexels_api_key is not None
    }
    
    return JSONResponse(
        content={
            "status": "healthy",
            "api_keys": api_keys_available,
            "active_processes": len(active_processes)
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Run with uvicorn
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )