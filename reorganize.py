#!/usr/bin/env python3
"""
Reorganize the project structure to follow the new modular design.
This script will move existing files into the correct directories or remove unnecessary files.
"""

import os
import shutil
from pathlib import Path

def ensure_dir(path):
    """Ensure a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def move_file(src, dest):
    """Move a file if it exists."""
    if os.path.exists(src):
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Moving {src} -> {dest}")
        shutil.move(src, dest)
    else:
        print(f"Skipping {src} (not found)")

def main():
    """Main function to reorganize the project."""
    # Ensure core structure
    for directory in [
        "core",
        "text",
        "video",
        "audio",
        "utils",
        "examples",
        "tests"
    ]:
        ensure_dir(directory)
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            print(f"Creating {init_file}")
            init_file.touch()
    
    # Ensure subdirectories
    for subdir in [
        "video/broll",
        "video/providers",
        "audio/extraction",
        "tests/text",
        "tests/video",
        "tests/audio",
        "tests/core"
    ]:
        ensure_dir(subdir)
        init_file = Path(subdir) / "__init__.py"
        if not init_file.exists():
            print(f"Creating {init_file}")
            init_file.touch()
    
    # Define file mappings (source -> destination)
    # None means the file will be deleted
    mappings = {
        # Core/main files to keep
        "main.py": "main.py",  # Keep in root
        "pipeline_heygen_voice.py": "examples/pipeline_heygen_voice.py",
        "pipeline_elevenlabs.py": "examples/pipeline_elevenlabs.py",
        "requirements.txt": "requirements.txt",  # Keep in root
        "LICENSE": "LICENSE",  # Keep in root
        "README.md": "README.md",  # Keep in root
        ".env": ".env",  # Keep in root
        "pyrightconfig.json": "pyrightconfig.json",  # Keep in root
        
        # Old files to be replaced by the new structure
        "pipeline_runner.py": None,  # Replaced by core/pipeline.py
        
        # Text module files
        "debug_segmentation.py": "text/debug_segmentation.py",
        "test_text_segmentaion.py": "tests/text/test_segmentation.py",
        
        # Video module files
        "merge_segments.py": "video/merge_segments.py",
        "merge_with_broll.py": "video/merge_with_broll.py",
        "simple_merge_broll.py": "video/simple_merge_broll.py",
        "simple_broll_merge.py": "video/simple_broll_merge.py",
        "test_moviepy.py": "tests/video/test_moviepy.py",
        
        # Audio module files
        "extract_broll_audio.py": "audio/extraction/extract_broll_audio.py",
        "silence_remover.py": "audio/silence_remover.py",
        
        # Broll module files
        "broll_generator.py": "video/broll/broll_generator.py",
        "create_pexels_broll.py": "video/broll/create_pexels_broll.py",
        "pexels_broll_fetcher.py": "video/broll/pexels_broll_fetcher.py",
        "run_broll_generator.py": "video/broll/run_broll_generator.py",
        "run_pexels.py": "video/broll/run_pexels.py",
        "test_broll.py": "tests/video/test_broll.py",
        "broll_text.txt": "video/broll/broll_text.txt",
        
        # Test files
        "test_setup.py": "tests/test_setup.py",
        
        # Miscellaneous files
        "resume_pipeline.py": "examples/resume_pipeline.py"
    }
    
    # Process the mappings
    for src, dest in mappings.items():
        if dest is None:
            if os.path.exists(src):
                print(f"Removing {src} (replaced)")
                os.remove(src)
        elif src != dest:  # Only move if source and destination are different
            move_file(src, dest)
    
    print("\nReorganization complete!")
    print("\nNext steps:")
    print("1. Update imports in moved files")
    print("2. Test the new structure with 'python main.py'")

if __name__ == "__main__":
    main() 