# AI Video Generation Pipeline

A modular, clean pipeline for generating AI videos from text using HeyGen avatars and Pexels B-roll footage.

## Features

- Convert text into AI-generated videos with talking avatars
- Automatically structure content with ChatGPT
- Use HeyGen for AI avatar generation
- Enhance videos with B-roll footage from Pexels
- API endpoint for video generation via HTTP requests
- Modular architecture for easy maintenance
- Support for parallel processing with multiple API keys
- Configurable video resolution, styling and formatting
- Automatic silence removal for better-paced videos

## Project Structure

The project follows a clean, modular structure:

```
.
├── core/                  # Core pipeline components
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   └── pipeline.py        # Main pipeline orchestration
├── text/
│   ├── __init__.py
│   └── processing.py      # Text processing (segmentation, clustering, ChatGPT)
├── video/
│   ├── __init__.py
│   ├── heygen.py          # HeyGen API client
│   ├── broll.py           # B-roll handling
│   └── merger.py          # Video merging
├── audio/
│   ├── __init__.py
│   ├── processing.py      # Audio generation and processing
│   └── scilence_remover.py # Silence removal for better pacing
├── api.py                 # FastAPI endpoint for HTTP access
├── main.py                # Command-line entry point
├── examples/              # Example usage
│   ├── example_usage.py   # Python API client example
│   └── curl_examples.sh   # cURL examples
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-video-generation.git
   cd ai-video-generation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API keys in a `.env` file:
   ```ini
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

### HTTP API Endpoint

The project includes a FastAPI endpoint for generating videos via HTTP requests. This enables integration with web applications or other services.

#### Running the API Server

```bash
python api.py
```

Or with Uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000, and interactive documentation at http://localhost:8000/docs.

#### API Endpoints

- `POST /generate-reel`: Start a new video generation process
- `GET /process/{process_id}`: Check the status of a process
- `GET /processes`: List all processes
- `GET /download/{process_id}`: Download the generated video
- `DELETE /process/{process_id}`: Delete a process and its files
- `GET /health`: Check API health status

#### Example API Request

```json
{
  "text": "Your text content here",
  "voice_id_1": "a426f8a763824ceaad3a2eb29c68e121",
  "look_id_1": "Raul_sitting_sofa_front_close",
  "look_id_2": "Raul_sitting_sofa_side_close",
  "remove_silence": true,
  "emotion": "Friendly",
  "background_color": "#008000"
}
```

#### Example Usage

See the `examples` directory for complete usage examples in Python and cURL.

Python example:
```python
# Simple API client example
import requests

# Start a process
response = requests.post("http://localhost:8000/generate-reel", json={
    "text": "Your text here",
    "voice_id_1": "your_voice_id",
    "look_id_1": "front_avatar_id",
    "look_id_2": "side_avatar_id"
})

process_id = response.json()["process_id"]

# Check status
status = requests.get(f"http://localhost:8000/process/{process_id}").json()
print(f"Status: {status['status']}")

# Download when complete
if status["status"] == "completed":
    video = requests.get(f"http://localhost:8000/download/{process_id}")
    with open("output.mp4", "wb") as f:
        f.write(video.content)
```

### Python API

You can also use the pipeline as a Python API:

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

## Pipeline Process

1. **Text Structuring**: Text is segmented and structured into video segments with shot types (front, side, broll) using ChatGPT.
2. **Video Generation**: Videos are generated with HeyGen for each segment.
3. **B-roll Processing**: If the segment is marked as B-roll, audio is extracted and used with video footage from Pexels.
4. **Video Merging**: All segments are merged into a final video with B-roll replacements as needed.
5. **Silence Removal**: Optionally remove silence from the final video to improve pacing.

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

## Requirements

- Python 3.8+
- FastAPI and Uvicorn (for API endpoint)
- API keys for:
  - HeyGen (for avatar generation)
  - OpenAI (for ChatGPT integration)
  - Pexels (for B-roll footage)
  - ElevenLabs (for TTS if using ElevenLabs)

## Extending the Pipeline

To extend the pipeline:

1. Add new functionality to the relevant module (text, video, audio)
2. Update the Pipeline class to use your new functionality
3. Add tests for your new functionality (if applicable)

## License

This project is licensed under a custom license - see the LICENSE file for details.