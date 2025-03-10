#!/usr/bin/env python3
"""
Test script for text segmentation and clustering.
This helps diagnose issues with chunk sizing and ensures we get the right number of chunks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules we want to test
from text.segmentation import split_text_into_chunks, get_chunk_durations, CHARS_PER_SECOND
from text.clustering import cluster_text_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_text")

def print_separator(title=""):
    print("\n" + "="*80)
    if title:
        print(f" {title} ")
        print("="*80)
    print()

def test_text_segmentation(text, expected_chunk_count=None):
    """Test the text segmentation function with the given text."""
    print_separator(f"Testing text segmentation")
    print(f"Input text ({len(text)} chars):")
    print(f"'{text[:100]}...'")
    
    # Split the text into chunks
    chunks = split_text_into_chunks(text)
    
    # Get durations for each chunk
    durations = get_chunk_durations(chunks)
    
    # Print results
    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        duration = durations[i]
        print(f"Chunk {i+1}: {len(chunk)} chars, {duration:.1f} seconds")
        print(f"  '{chunk[:50]}...'")
    
    # Check if we got the expected number of chunks
    if expected_chunk_count is not None:
        if len(chunks) == expected_chunk_count:
            print(f"\n✅ Got expected {expected_chunk_count} chunks")
        else:
            print(f"\n❌ Expected {expected_chunk_count} chunks, got {len(chunks)}")
    
    # Check if chunks are in our target range
    target_min = 9
    target_max = 11
    out_of_range = [i for i, d in enumerate(durations) if d < target_min or d > target_max]
    
    if not out_of_range:
        print(f"✅ All chunks are within target range ({target_min}-{target_max} seconds)")
    else:
        print(f"❌ {len(out_of_range)} chunks are outside target range:")
        for i in out_of_range:
            print(f"  Chunk {i+1}: {durations[i]:.1f} seconds")
    
    return chunks

def test_text_clustering(chunks):
    """Test the text clustering function with the given chunks."""
    print_separator(f"Testing text clustering")
    print(f"Input: {len(chunks)} chunks")
    
    # Cluster the chunks
    clusters = cluster_text_chunks(chunks, group_size=1)  # Use group_size=1 to test individual chunks
    
    # Print results
    print(f"\nCreated {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}:")
        print(f"  Original position: {cluster['original_order']}")
        print(f"  Chunk range: {cluster['start_chunk']}-{cluster['end_chunk']}")
        print(f"  Text length: {cluster['char_length']} chars")
        estimated_seconds = cluster['char_length'] / CHARS_PER_SECOND
        print(f"  Estimated duration: {estimated_seconds:.1f} seconds")
        print(f"  Text: '{cluster['text'][:50]}...'")
    
    return clusters

def test_with_sample_romanian_text():
    """Test with a sample of Romanian text similar to what the user provided."""
    sample_text = """
        Distracția este una dintre cele mai mari plăceri ale vieții! Indiferent 
        dacă joci sportul tău preferat, petreci timp cu prietenii și familia sau 
        doar te relaxezi cu o carte bună, distracția poate lua multe forme. 
        Unii oameni găsesc bucurie în activități creative, precum pictura sau muzica, 
        în timp ce alții preferă aventurile în aer liber, cum ar fi drumețiile sau înotul. 
        Cheia este să descoperi ce te face să zâmbești și să te bucuri pe deplin de acele 
        momente. Nu uita să râzi, să dansezi și să te joci – aceste gesturi simple pot 
        însenina chiar și cele mai întunecate zile. Viața e prea scurtă pentru a fi mereu 
        serioși, așa că asigură-te că îți faci loc pentru bucurie și joacă în rutina ta zilnică.
    """
    
    # Based on the text length, we'd expect around 4-5 chunks of 9-11 seconds each
    expected_chunks = 5
    
    chunks = test_text_segmentation(sample_text, expected_chunks)
    clusters = test_text_clustering(chunks)
    
    # Check total estimated duration of all chunks
    total_chars = sum(len(chunk) for chunk in chunks)
    total_duration = total_chars / CHARS_PER_SECOND
    print(f"\nTotal estimated duration: {total_duration:.1f} seconds")
    print(f"Average chunk duration: {total_duration/len(chunks):.1f} seconds")

def test_with_very_long_text():
    """Test with a very long paragraph to ensure proper splitting."""
    very_long_text = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.
    """
    
    chunks = test_text_segmentation(very_long_text)
    clusters = test_text_clustering(chunks)

def test_with_very_short_text():
    """Test with a very short text to ensure proper handling."""
    short_text = "This is a very short text that should only make one chunk."
    
    chunks = test_text_segmentation(short_text, 1)
    clusters = test_text_clustering(chunks)

def debug_clustering_pipeline(text, group_size=1):
    """Debug the entire clustering pipeline with the given text."""
    print_separator("DEBUGGING FULL PIPELINE")
    print(f"Input text: {len(text)} chars")
    print(f"Group size: {group_size}")
    
    # First get the chunks
    chunks = split_text_into_chunks(text)
    
    # Cluster with the specified group size
    clusters = cluster_text_chunks(chunks, group_size=group_size)
    
    # Print detailed information about each cluster
    print(f"\nCreated {len(clusters)} clusters with group_size={group_size}:")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1}:")
        print(f"  Original position: {cluster['original_order']}")
        print(f"  Chunk range: {cluster['start_chunk']}-{cluster['end_chunk']}")
        print(f"  Number of chunks: {cluster['num_chunks']}")
        print(f"  Text length: {cluster['char_length']} chars")
        estimated_seconds = cluster['char_length'] / CHARS_PER_SECOND
        print(f"  Estimated duration: {estimated_seconds:.1f} seconds")
        print(f"  Text: '{cluster['text'][:50]}...'")
    
    # Summarize
    total_chars = sum(cluster['char_length'] for cluster in clusters)
    total_duration = total_chars / CHARS_PER_SECOND
    print(f"\nTotal estimated duration: {total_duration:.1f} seconds")
    print(f"Average cluster duration: {total_duration/len(clusters):.1f} seconds")
    
    return clusters
    
if __name__ == "__main__":
    print_separator("TEXT SEGMENTATION TESTS")
    print("Testing different scenarios to diagnose segmentation issues")
    
    test_with_sample_romanian_text()
    
    # Get user input for testing with specific text
    print_separator("Custom Test")
    print("Would you like to test with custom text? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        print("\nEnter text to test (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if not line and lines and not lines[-1]:
                break
            lines.append(line)
        
        custom_text = "\n".join(lines)
        print("\nEnter group size for clustering (default 1):")
        try:
            group_size = int(input().strip() or "1")
        except ValueError:
            group_size = 1
        
        debug_clustering_pipeline(custom_text, group_size)
    
    print("\nTests completed.")