#!/usr/bin/env python3
"""
A simple test script to verify the environment setup.
"""

import os
import sys

def check_environment():
    """Check if the environment is set up correctly."""
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if required modules are installed
    try:
        import numpy
        print(f"NumPy version: {numpy.__version__}")
        
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        import moviepy.editor
        print(f"MoviePy is installed")
        
        import nltk
        print(f"NLTK version: {nltk.__version__}")
        
        # Check if .env file exists
        if os.path.exists('.env'):
            print(".env file exists")
        else:
            print("Warning: .env file not found")
        
        # Check if required directories exist
        for directory in ['audio', 'video', 'text', 'utils', 'output']:
            if os.path.isdir(directory):
                print(f"'{directory}' directory exists")
            else:
                print(f"Warning: '{directory}' directory not found")
        
        print("\nEnvironment setup looks good! You can start running the pipeline.")
        print("To run the ElevenLabs pipeline: python pipeline_elevenlabs.py")
        print("To run the HeyGen voice pipeline: python pipeline_heygen_voice.py")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Some required packages are not installed. Try running:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    print("Checking environment setup...")
    check_environment() 