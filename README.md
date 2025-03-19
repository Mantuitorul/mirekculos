# Video Generation Pipeline

A modular, clean pipeline for generating AI videos from text using HeyGen avatars and Pexels B-roll footage.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Development](#development)

## Features

- Convert text into AI-generated videos with talking avatars
- Automatically structure content with ChatGPT
- Use HeyGen for AI avatar generation
- Enhance videos with B-roll footage from Pexels
- Modular architecture with clean separation of concerns
- Support for parallel processing with multiple API keys
- Configurable video resolution, styling and formatting

## Project Structure

The project has been refactored to follow a clean, modular structure:

```
.
├── core/                  # Core pipeline components
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management
│   └── pipeline.py        # Main pipeline orchestration
├── text/                  # Text processing components
│   ├── chatgpt_integration.py  # ChatGPT integration for structuring
│   ├── segmentation.py    # Text segmentation utilities
│   └── clustering.py      # Text clustering utilities
├── video/                 # Video processing components
│   ├── heygen_client.py   # HeyGen API client
│   └── merger.py          # Video merging utilities
├── audio/                 # Audio processing components
├── utils/                 # Common utilities
├── main.py                # Main command-line entry point
├── pipeline_heygen_voice.py  # Legacy entry point
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/video-generation.git
   cd video-generation
   ```

2. Create a virtual environment (Python 3.8+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API keys in a `.env` file:
   ```
   HEYGEN_API_KEY=your_heygen_api_key
   OPENAI_API_KEY=your_openai_api_key
   PEXELS_API_KEY=your_pexels_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   
   # For multiple HeyGen keys:
   HEYGEN_API_KEY_1=your_first_heygen_key
   HEYGEN_API_KEY_2=your_second_heygen_key
   ```

## Usage

### Command-line Interface

```bash
python main.py --text "Your text here" --front-avatar "avatar_id_1" --side-avatar "avatar_id_2"
```

#### Options:

- `--text`: Input text for video generation
- `--text-file`: File containing input text (alternative to `--text`)
- `--front-avatar`: HeyGen avatar ID for front shots (required)
- `--side-avatar`: HeyGen avatar ID for side shots (required)
- `--voice-id`: HeyGen voice ID (optional)
- `--emotion`: Voice emotion (optional, choices: "Excited", "Friendly", "Serious", "Soothing", "Broadcaster")
- `--background`: Background color (hex, default: "#008000")
- `--width`: Video width in pixels (default: 720)
- `--height`: Video height in pixels (default: 1280)
- `--output`: Output filename (default: "final_output.mp4")
- `--debug`: Enable debug mode
- `--debug-dir`: Debug output directory (default: "debug_output")

### Python API

```python
import asyncio
from core import Pipeline

async def generate_video():
    pipeline = Pipeline(
        width=720,
        height=1280,
        debug_mode=True
    )
    
    result = await pipeline.run(
        text="Your text here",
        front_avatar_id="your_front_avatar_id",
        side_avatar_id="your_side_avatar_id",
        heygen_voice_id="your_voice_id",
        heygen_emotion="Friendly",
        output_filename="output.mp4"
    )
    
    print(f"Video generated: {result['final_video']}")

asyncio.run(generate_video())
```

## Configuration

The pipeline uses a centralized configuration system. You can customize configuration by creating a `Config` object:

```python
from core import Config

# Create a custom config (uses .env by default)
config = Config()

# Access API keys
heygen_keys = config.heygen_api_keys
openai_key = config.openai_api_key

# Create a pipeline with this config
pipeline = Pipeline(config=config)
```

## Examples

See `pipeline_heygen_voice.py` for a full working example.

## Development

To extend the pipeline:

1. Add new functionality to the relevant module (text, video, audio)
2. Update the Pipeline class to use your new functionality
3. Add tests for your new functionality

### Adding New AI Providers

To add a new AI provider:

1. Create a new client in the appropriate module
2. Implement required methods for integration
3. Update the Pipeline class to conditionally use your provider

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.