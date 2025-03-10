#!/usr/bin/env python3
# audio/clustering.py
"""
Audio clustering utilities.
Groups audio chunks into larger clusters for processing.
"""

import logging
from typing import List
from pydub import AudioSegment

# Configure logging
logger = logging.getLogger(__name__)

def cluster_audio(chunks: List[AudioSegment], group_size: int = 3) -> List[AudioSegment]:
    """
    Clusters a list of audio chunks into groups of maximum `group_size` chunks.
    Each group is concatenated into a single AudioSegment.
    
    Args:
        chunks: List of audio chunks
        group_size: Maximum number of chunks in each cluster
        
    Returns:
        A list of concatenated audio clusters
    """
    logger.info(f"Clustering {len(chunks)} audio chunks with max group size {group_size}")
    
    if not chunks:
        logger.warning("No chunks to cluster, returning empty list")
        return []
        
    clusters = []
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i+group_size]
        if group:
            # Concatenate the group
            cluster = group[0]
            for chunk in group[1:]:
                cluster += chunk
                
            cluster_length = len(cluster) / 1000  # Convert to seconds
            clusters.append(cluster)
            
            logger.info(f"Created cluster {len(clusters)} with {len(group)} chunks, length: {cluster_length:.2f}s")
    
    logger.info(f"Created {len(clusters)} clusters in total")
    return clusters