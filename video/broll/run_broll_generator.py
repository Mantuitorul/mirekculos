#!/usr/bin/env python3
# run_broll_generator.py
"""
Run only the broll generator step using the broll_generator.py module.
This approach uses MoviePy for better video processing and is more reliable.
"""

import asyncio
import logging
from pathlib import Path
from pipeline_runner import run_broll_generator_step, ensure_output_dir

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default output directory
OUTPUT_DIR = Path("output")

async def run_broll_generator():
    """
    Run the broll generator step.
    """
    logger.info("Starting broll generator")
    
    # Setup output directory
    ensure_output_dir(OUTPUT_DIR)
    
    # Run the broll generator step
    results = await run_broll_generator_step()
    
    if results:
        logger.info(f"Broll generation completed with {results['success']} successful and {results['failed']} failed segments")
    else:
        logger.error("Broll generation failed")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_broll_generator()) 