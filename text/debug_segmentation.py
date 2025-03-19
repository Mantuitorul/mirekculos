#!/usr/bin/env python3
"""
Debug script for text segmentation and clustering.
This helps identify issues with the segmentation process and provides immediate feedback.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("debug")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the modules
from text.segmentation import split_text_into_chunks, get_chunk_durations, CHARS_PER_SECOND
from text.clustering import cluster_text_chunks

def debug_process(input_text, cluster_size=1):
    """Process input text through the segmentation and clustering pipeline."""
    print("\n" + "="*80)
    print(" TEXT SEGMENTATION DEBUG ")
    print("="*80)
    
    # Step 1: Segment the text into chunks
    print("\nStep 1: Segmenting text into chunks...")
    chunks = split_text_into_chunks(input_text)
    
    # Get estimated durations
    durations = get_chunk_durations(chunks)
    
    # Print chunk details
    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        duration = durations[i]
        print(f"\nChunk {i+1} ({duration:.1f}s):")
        print(f"  '{chunk}'")
    
    # Check for chunks outside target range
    target_min = 9
    target_max = 11
    out_of_range = [(i, d) for i, d in enumerate(durations) if d < target_min or d > target_max]
    
    if out_of_range:
        print(f"\n⚠️ WARNING: Found {len(out_of_range)} chunks outside target range:")
        for i, duration in out_of_range:
            print(f"  Chunk {i+1}: {duration:.1f}s {'(too short)' if duration < target_min else '(too long)'}")
    else:
        print(f"\n✅ All chunks are within target range ({target_min}-{target_max}s)")
    
    # Step 2: Clustering chunks
    print(f"\nStep 2: Clustering chunks with group_size={cluster_size}...")
    clusters = cluster_text_chunks(chunks, group_size=cluster_size)
    
    print(f"\nCreated {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        cluster_duration = cluster["char_length"] / CHARS_PER_SECOND
        print(f"\nCluster {i+1} ({cluster_duration:.1f}s):")
        print(f"  Original position: {cluster['original_order']}")
        print(f"  Chunks: {cluster['start_chunk']}-{cluster['end_chunk']} ({cluster['num_chunks']} chunks)")
        print(f"  Text: '{cluster['text']}'")
        
        # Check if cluster exceeds target duration
        if cluster_duration > target_max:
            print(f"  ⚠️ WARNING: Cluster exceeds maximum target duration ({target_max}s)")
        elif cluster_duration < target_min:
            print(f"  ⚠️ WARNING: Cluster is below minimum target duration ({target_min}s)")
    
    # Summary
    print("\n" + "="*80)
    print(" SUMMARY ")
    print("="*80)
    
    total_duration = sum(durations)
    print(f"\nText statistics:")
    print(f"  Total length: {len(input_text)} characters")
    print(f"  Estimated total duration: {total_duration:.1f} seconds")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Average chunk duration: {total_duration/len(chunks):.1f} seconds")
    print(f"  Number of clusters: {len(clusters)}")
    
    # Final recommendation
    if out_of_range:
        print("\n⚠️ RECOMMENDATION: Text segmentation needs tuning. Some chunks are outside target range.")
        if cluster_size > 1:
            print("  Try using cluster_size=1 to ensure each video segment is 9-11 seconds.")
    else:
        print("\n✅ Text segmentation looks good. All chunks are within target range.")
    
    return chunks, clusters

def main():
    print("Text Segmentation Debug Tool")
    print("This tool helps diagnose issues with text segmentation for HeyGen videos.")
    
    # Determine input method
    print("\nHow would you like to input text?")
    print("1. Enter text directly")
    print("2. Read from a file")
    choice = input("Enter choice (1 or 2): ").strip()
    
    # Get the input text
    if choice == "2":
        filename = input("\nEnter path to text file: ").strip()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                input_text = f.read()
            print(f"Read {len(input_text)} characters from {filename}")
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("\nEnter or paste your text below.")
        print("Press Ctrl+D (Unix) or Ctrl+Z (Windows) followed by Enter when finished.")
        
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        input_text = "\n".join(lines)
    
    # Get clustering parameters
    try:
        cluster_size = int(input("\nEnter cluster size (1 recommended): ").strip() or "1")
    except ValueError:
        cluster_size = 1
        print("Invalid input. Using default cluster size of 1.")
    
    # Process the text
    chunks, clusters = debug_process(input_text, cluster_size)
    
    print("\nDebug complete.")

if __name__ == "__main__":
    main()