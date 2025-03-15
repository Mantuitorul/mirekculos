#!/usr/bin/env python3
"""
Standalone script to test B-roll processing on an existing video file.
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Setup explicit NLTK download before importing our modules
print("Setting up NLTK resources...")
try:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK resources downloaded successfully")
except Exception as e:
    print(f"Warning: NLTK resource download issue (non-critical): {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Function to detect language
def detect_language(text):
    """Simple language detection based on character usage"""
    # Check for Romanian-specific characters
    romanian_chars = set('ăâîșțéáóúí')
    text_lower = text.lower()
    
    # Count Romanian characters
    romanian_char_count = sum(1 for c in text_lower if c in romanian_chars)
    
    if romanian_char_count > 0:
        return "romanian"
    return "english"

async def process_existing_video(
    video_path: str,
    text: str,
    output_dir: str,
    output_filename: str = "video_with_broll_n.mp4",
    num_broll: int = 3,
    broll_duration: float = 5.0,
    api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    enhance_queries: bool = True,
    language: Optional[str] = None
):
    """
    Process an existing video file with B-roll.
    
    Args:
        video_path: Path to the existing video
        text: Original text content (for keyword extraction)
        output_dir: Directory for output and temp files
        output_filename: Name of the output file
        num_broll: Number of B-roll segments to insert
        broll_duration: Duration of each B-roll segment
        api_key: Pexels API key (optional)
        openai_api_key: OpenAI API key for query enhancement (optional)
        enhance_queries: Whether to use OpenAI to enhance search queries
        language: Language of the text (optional, auto-detected if not provided)
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Detect language if not provided
    if not language:
        language = detect_language(text)
        logger.info(f"Detected language: {language}")
    
    # Load environment variables if API key not provided
    if not api_key:
        load_dotenv()
        api_key = os.getenv("PEXELS_API_KEY")
        if not api_key:
            logger.warning("No Pexels API key found! Set PEXELS_API_KEY in .env file or pass with --api-key")
    
    # Use OpenAI API key from env var if not provided
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key and enhance_queries:
            logger.warning("No OpenAI API key found! Set OPENAI_API_KEY in .env file or pass with --openai-api-key. Query enhancement will be disabled.")
            enhance_queries = False
    
    # Import module here to ensure NLTK is initialized first
    try:
        from post_processing import apply_broll_post_processing
    except ImportError as e:
        logger.error(f"Failed to import post_processing module: {str(e)}")
        print("Make sure the post_processing module is in your Python path")
        return
    
    # Process the video
    try:
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output will be saved to: {output_path / output_filename}")
        
        result_path = await apply_broll_post_processing(
            video_path_or_files=video_path,
            text=text,
            output_dir=output_path,
            output_filename=output_filename,
            num_broll=num_broll,
            broll_duration=broll_duration,
            api_key=api_key,
            openai_api_key=openai_api_key,
            enhance_queries=enhance_queries
        )
        
        logger.info(f"Processing complete! Video saved to: {result_path}")
        return result_path
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Parse arguments and run the script"""
    parser = argparse.ArgumentParser(description="Apply B-roll processing to an existing video file")
    
    parser.add_argument(
        "--video", "-v", 
        required=True,
        help="Path to the input video file"
    )
    
    parser.add_argument(
        "--text", "-t", 
        help="Original text content (for keyword extraction)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for output and temporary files"
    )
    
    parser.add_argument(
        "--output-filename", "-f",
        default="video_with_broll.mp4",
        help="Name of the output video file"
    )
    
    parser.add_argument(
        "--num-broll", "-n",
        type=int,
        default=3,
        help="Number of B-roll segments to insert"
    )
    
    parser.add_argument(
        "--broll-duration", "-d",
        type=float,
        default=5.0,
        help="Duration of each B-roll segment in seconds"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        help="Pexels API key (defaults to PEXELS_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--openai-api-key", "-ok",
        help="OpenAI API key for query enhancement (defaults to OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--enhance-queries", "-eq",
        action="store_true",
        default=True,
        help="Use OpenAI to enhance search queries (default: True)"
    )
    
    parser.add_argument(
        "--no-enhance-queries", "-neq",
        action="store_false",
        dest="enhance_queries",
        help="Disable OpenAI query enhancement"
    )
    
    parser.add_argument(
        "--text-file", "-tf",
        help="Text file containing the original content (alternative to --text)"
    )
    
    parser.add_argument(
        "--language", "-l",
        help="Language of the text (english, romanian, etc.)"
    )
    
    args = parser.parse_args()
    
    # Ensure we have either text or text-file
    if not args.text and not args.text_file:
        print("Error: You must provide either --text or --text-file")
        parser.print_help()
        sys.exit(1)
    
    # Load text from file if specified
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            return
    else:
        text = args.text
    
    # Run the processing
    asyncio.run(process_existing_video(
        video_path=args.video,
        text=text,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        num_broll=args.num_broll,
        broll_duration=args.broll_duration,
        api_key=args.api_key,
        openai_api_key=args.openai_api_key,
        enhance_queries=args.enhance_queries,
        language=args.language
    ))

if __name__ == "__main__":
    main()