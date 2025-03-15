#!/usr/bin/env python3
# text/segmentation.py
"""
Text segmentation utilities.
Splits text into natural chunks for processing by HeyGen's voice API.
"""

import logging
from typing import List, Tuple
import re

# Configure logging
logger = logging.getLogger(__name__)

# Constants for estimating speech duration
# Average reading speed is about 150 words per minute or 2.5 words per second
# Assuming average word length of 5 characters + 1 space = 6 characters per word
# This gives us roughly 15 characters per second
CHARS_PER_SECOND = 15
TARGET_MIN_SECONDS = 4  # Minimum target seconds per chunk
TARGET_MAX_SECONDS = 7  # Maximum target seconds per chunk
DEFAULT_TARGET_SECONDS = 10  # Default target for function parameter

def split_text_into_chunks(text: str, target_seconds_per_chunk: int = DEFAULT_TARGET_SECONDS) -> List[str]:
    """
    Split text into natural chunks based on sentence boundaries,
    targeting chunks that would be approximately 9-11 seconds when spoken.
    This creates the optimal pacing for video editing with 3-4 switches
    in a 40-second video.
    
    Args:
        text: The input text to split
        target_seconds_per_chunk: Target duration in seconds for each chunk
        
    Returns:
        List of text chunks optimized for the specified duration range
    """
    # Estimate characters per chunk based on target duration
    min_chars_per_chunk = TARGET_MIN_SECONDS * CHARS_PER_SECOND  # 135 chars for 9 seconds
    max_chars_per_chunk = TARGET_MAX_SECONDS * CHARS_PER_SECOND  # 165 chars for 11 seconds
    target_chars_per_chunk = target_seconds_per_chunk * CHARS_PER_SECOND  # Default 150 chars for 10 seconds
    
    logger.info(f"Targeting chunks between {TARGET_MIN_SECONDS}-{TARGET_MAX_SECONDS} seconds")
    logger.info(f"Character range: {min_chars_per_chunk}-{max_chars_per_chunk} chars")
    
    # Clean up text (remove multiple spaces, etc.)
    text = ' '.join(text.split())
    
    # Split on sentence boundaries (., !, ?)
    # This regex matches sentence-ending punctuation followed by a space or newline
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Make sure sentences end with punctuation
    for i in range(len(sentences)):
        if i < len(sentences) - 1 and not sentences[i].rstrip().endswith(('.', '!', '?')):
            sentences[i] = sentences[i] + '.'
    
    logger.info(f"Split text into {len(sentences)} sentences")
    
    # First pass: Build preliminary chunks
    preliminary_chunks = []
    current_chunk = ""
    current_chunk_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        sentence_duration = sentence_size / CHARS_PER_SECOND
        
        # If adding this sentence would exceed the max chunk size, start a new chunk
        if current_chunk_size + sentence_size > max_chars_per_chunk and current_chunk:
            preliminary_chunks.append(current_chunk.strip())
            current_chunk = ""
            current_chunk_size = 0
        
        # If a single sentence is longer than max_chars_per_chunk, we need to split it
        if sentence_size > max_chars_per_chunk:
            # If there's content in the current chunk, finalize it first
            if current_chunk:
                preliminary_chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_size = 0
            
            # Split the long sentence into smaller pieces
            # First try by phrases
            phrase_splits = re.split(r'(?<=[,;:])\s+', sentence)
            
            if len(phrase_splits) > 1:
                # If we can split by phrases, do that
                phrase_chunk = ""
                phrase_chunk_size = 0
                
                for phrase in phrase_splits:
                    phrase_size = len(phrase)
                    
                    # If adding this phrase would exceed max_chars_per_chunk, start a new phrase chunk
                    if phrase_chunk_size + phrase_size > max_chars_per_chunk and phrase_chunk:
                        preliminary_chunks.append(phrase_chunk.strip())
                        phrase_chunk = phrase
                        phrase_chunk_size = phrase_size
                    else:
                        if phrase_chunk:
                            phrase_chunk += " " + phrase
                            phrase_chunk_size += phrase_size + 1  # +1 for the space
                        else:
                            phrase_chunk = phrase
                            phrase_chunk_size = phrase_size
                
                # Don't forget the last phrase chunk
                if phrase_chunk:
                    if phrase_chunk_size < min_chars_per_chunk and preliminary_chunks:
                        # If the phrase is too short, try to merge with the previous chunk
                        last_chunk = preliminary_chunks[-1]
                        combined_size = len(last_chunk) + phrase_chunk_size + 1  # +1 for space
                        
                        if combined_size <= max_chars_per_chunk:
                            preliminary_chunks[-1] = last_chunk + " " + phrase_chunk
                        else:
                            preliminary_chunks.append(phrase_chunk.strip())
                    else:
                        preliminary_chunks.append(phrase_chunk.strip())
            else:
                # As a last resort for very long sentences with no phrase breaks,
                # split by word count to get as close to target duration as possible
                words = sentence.split()
                word_chunk = ""
                word_chunk_size = 0
                target_word_count = int(max_chars_per_chunk / 6)  # Approximate words per chunk
                
                for i in range(0, len(words), target_word_count):
                    group = words[i:i+target_word_count]
                    word_chunk = " ".join(group)
                    word_chunk_size = len(word_chunk)
                    
                    if word_chunk_size > min_chars_per_chunk:
                        preliminary_chunks.append(word_chunk.strip())
                    elif preliminary_chunks:
                        # If too short, try to merge with previous chunk
                        last_chunk = preliminary_chunks[-1]
                        combined_size = len(last_chunk) + word_chunk_size + 1
                        
                        if combined_size <= max_chars_per_chunk:
                            preliminary_chunks[-1] = last_chunk + " " + word_chunk
                        else:
                            preliminary_chunks.append(word_chunk.strip())
                    else:
                        preliminary_chunks.append(word_chunk.strip())
        else:
            # For regular sentences, add to current chunk
            new_chunk_size = current_chunk_size + sentence_size
            if current_chunk:
                new_chunk_size += 1  # Account for space
            
            # Check if adding this sentence would push us beyond max_chars_per_chunk
            if new_chunk_size > max_chars_per_chunk:
                preliminary_chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_chunk_size = sentence_size
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_chunk_size = new_chunk_size
    
    # Don't forget the last chunk
    if current_chunk:
        preliminary_chunks.append(current_chunk.strip())
    
    # Second pass: Optimize chunk sizes
    chunks = []
    i = 0
    
    while i < len(preliminary_chunks):
        chunk = preliminary_chunks[i]
        chunk_size = len(chunk)
        duration = chunk_size / CHARS_PER_SECOND
        
        # If chunk is too short and not the last one, try to merge with next chunk
        if duration < TARGET_MIN_SECONDS and i < len(preliminary_chunks) - 1:
            next_chunk = preliminary_chunks[i + 1]
            combined_size = chunk_size + len(next_chunk) + 1  # +1 for space
            combined_duration = combined_size / CHARS_PER_SECOND
            
            # If merging would keep us within max duration, do it
            if combined_duration <= TARGET_MAX_SECONDS:
                chunks.append(chunk + " " + next_chunk)
                i += 2  # Skip the next chunk since we merged it
                logger.debug(f"Merged chunks {i-1} and {i} to get {combined_duration:.1f}s")
                continue
        
        # If we get here, just add the chunk as is
        chunks.append(chunk)
        i += 1
    
    # Final validation
    for i, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        duration = chunk_size / CHARS_PER_SECOND
        
        if duration < TARGET_MIN_SECONDS:
            logger.warning(f"Chunk {i+1} is shorter than target minimum: {duration:.1f}s < {TARGET_MIN_SECONDS}s")
        elif duration > TARGET_MAX_SECONDS:
            logger.warning(f"Chunk {i+1} is longer than target maximum: {duration:.1f}s > {TARGET_MAX_SECONDS}s")
            
            # If it's extremely long, try one more split
            if duration > TARGET_MAX_SECONDS + 2:  # More than 2 seconds over
                logger.warning(f"Attempting emergency split of overly long chunk {i+1}")
                
                # Try to find a good breaking point near the middle
                midpoint = len(chunk) // 2
                left_part = chunk[:midpoint].rstrip()
                right_part = chunk[midpoint:].lstrip()
                
                # If we have a clean break point, replace this chunk with two chunks
                if left_part and right_part:
                    chunks[i] = left_part
                    chunks.insert(i+1, right_part)
                    logger.info(f"Split chunk {i+1} into two chunks")
    
    # Log the final chunks with their durations
    logger.info(f"Created {len(chunks)} text chunks after optimization")
    for i, chunk in enumerate(chunks):
        est_duration = len(chunk) / CHARS_PER_SECOND
        logger.info(f"Chunk {i+1}: {len(chunk)} chars, est. duration: {est_duration:.1f}s")
        if est_duration < TARGET_MIN_SECONDS or est_duration > TARGET_MAX_SECONDS:
            logger.warning(f"⚠️ Chunk {i+1} duration ({est_duration:.1f}s) is outside target range")
    
    return chunks

def get_chunk_durations(chunks: List[str]) -> List[float]:
    """
    Estimate the duration of each text chunk in seconds.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of estimated durations in seconds
    """
    return [len(chunk) / CHARS_PER_SECOND for chunk in chunks]