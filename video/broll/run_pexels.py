#!/usr/bin/env python3
# run_pexels.py
"""
Run only the Pexels broll creation step using the processing_results.json file.
"""

import asyncio
import logging
import json
from pathlib import Path
from pipeline_runner import create_pexels_broll, ensure_output_dir

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default output directory
OUTPUT_DIR = Path("output")

async def run_pexels_from_processing_results():
    """
    Run only the Pexels step (step 4) to create broll videos using processing_results.json.
    """
    logger.info("Running just the Pexels broll video creation step")
    
    # Setup output directory
    output_dir = ensure_output_dir(OUTPUT_DIR)
    
    # Load segment results from processing_results.json
    processing_results_file = output_dir / "processing_results.json"
    if not processing_results_file.exists():
        raise ValueError("processing_results.json not found, cannot run Pexels step")
    
    with open(processing_results_file, "r", encoding="utf-8") as f:
        segments_results = json.load(f)
    
    logger.info(f"Loaded results from processing_results.json")
    
    # Run the Pexels broll creation step
    await create_pexels_broll(segments_results)
    
    logger.info("Pexels broll creation step completed")
    
    # Return the list of broll segments
    return [s for s in segments_results if s.get("is_broll", False)]

if __name__ == "__main__":
    asyncio.run(run_pexels_from_processing_results()) 