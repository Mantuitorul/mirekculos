#!/usr/bin/env python3
# text/clustering.py
"""
Text clustering utilities.
Groups text chunks into larger clusters for processing.
"""

import logging
from typing import List, Dict, Any
from text.segmentation import CHARS_PER_SECOND

# Configure logging
logger = logging.getLogger(__name__)

def cluster_text_chunks(chunks: List[str], group_size: int = 1) -> List[Dict[str, Any]]:
    """
    Clusters a list of text chunks into groups of maximum `group_size` chunks.
    Each group is concatenated into a single string with appropriate spacing.
    Preserves the original ordering information.
    
    IMPORTANT: For proper timing of 9-11 seconds per video segment, use group_size=1
    
    Args:
        chunks: List of text chunks
        group_size: Maximum number of chunks in each cluster (use 1 for 9-11 second chunks)
        
    Returns:
        A list of dictionaries with cluster text and metadata including original positions
    """
    logger.info(f"Clustering {len(chunks)} text chunks with max group size {group_size}")
    
    if not chunks:
        logger.warning("No chunks to cluster, returning empty list")
        return []
    
    # Verify all chunks are within proper duration range before clustering
    for i, chunk in enumerate(chunks):
        chunk_duration = len(chunk) / CHARS_PER_SECOND
        if chunk_duration < 9 or chunk_duration > 11:
            logger.warning(f"Chunk {i} has estimated duration of {chunk_duration:.1f}s, "
                          f"outside target range of 9-11 seconds")
    
    clusters = []
    for i in range(0, len(chunks), group_size):
        # Get the current group of chunks
        group = chunks[i:i+group_size]
        
        if group:
            # Record the original positions and join text with spaces
            start_pos = i
            end_pos = i + len(group) - 1
            cluster_text = " ".join(group)
            cluster_duration = len(cluster_text) / CHARS_PER_SECOND
            
            # Create cluster with metadata
            cluster = {
                "text": cluster_text,
                "start_chunk": start_pos,
                "end_chunk": end_pos,
                "num_chunks": len(group),
                "char_length": len(cluster_text),
                "duration": cluster_duration,
                "original_order": len(clusters)  # Keep track of original order
            }
            
            clusters.append(cluster)
            logger.info(f"Created cluster {len(clusters)} with chunks {start_pos}-{end_pos}, "
                       f"length: {len(cluster_text)} chars, duration: {cluster_duration:.1f}s")
    
    # Verify the final clusters
    for i, cluster in enumerate(clusters):
        if cluster["duration"] > 11:
            logger.warning(f"⚠️ Cluster {i} has estimated duration of {cluster['duration']:.1f}s, "
                          f"which exceeds the target maximum of 11 seconds")
    
    logger.info(f"Created {len(clusters)} text clusters in total")
    return clusters