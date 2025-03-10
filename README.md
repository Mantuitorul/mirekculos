# Text-to-Video Pipeline

A modular, well-structured pipeline that converts text to high-quality videos with animated avatars.

## Overview

This project provides an end-to-end pipeline that:

1. Converts text to speech (using either ElevenLabs API or HeyGen's voice API)
2. Intelligently segments text or audio into natural chunks
3. Generates videos with animated avatars using HeyGen API, alternating between different poses
4. Merges everything into a seamless final video
5. Optionally adds relevant B-roll footage using the Pexels API

## Project Structure

The project follows a modular design with clear separation of concerns:

```
project/
├── pipeline_elevenlabs.py  # Entry point for ElevenLabs-based pipeline
├── pipeline_heygen_voice.py # Entry point for HeyGen voice-based pipeline
├── pipeline_runner.py      # Common pipeline orchestrator
├── audio/                  # Audio processing modules
│   ├── generation.py       # Text-to-speech conversion
│   ├── silence_split.py    # Audio silence detection
│   ├── clustering.py       # Audio chunk clustering
│   └── upload.py           # Audio file uploading
├── video/                  # Video processing modules
│   ├── heygen_client.py    # HeyGen API client
│   └── merger.py           # Video merging utilities
├── text/                   # Text processing modules
│   ├── segmentation.py     # Text segmentation
│   └── clustering.py       # Text chunk clustering
└── utils/                  # Utility modules
    └── config.py           # Configuration management
```

## Requirements

```bash
python -m pip install -r requirements.txt
```

## API Keys

You need to obtain the following API keys:

1. **HeyGen API Key**: For avatar video generation (required)
2. **ElevenLabs API Key**: For high-quality text-to-speech (required only for ElevenLabs mode)
3. **Pexels API Key**: For B-roll footage (required only when using B-roll feature)

Store these in a `.env` file in the project root:

```
HEYGEN_API_KEY=your_heygen_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
PEXELS_API_KEY=your_pexels_api_key
```

## Usage

### Quick Start

The project provides two separate entry points for the different voice generation methods:

#### Using HeyGen's Voice API

Edit `pipeline_heygen_voice.py` with your parameters:

```python
# Set your parameters here
TEXT = "Your text to convert to video."
FRONT_MODEL_ID = "Vernon_standing_lounge_front"
SIDE_MODEL_ID = "Vernon_standing_lounge_side"

# Required: HeyGen voice ID
HEYGEN_VOICE_ID = "b5db595bfd744dfc9a2087c5822cf29b"

# Optional: HeyGen voice emotion
# Options: 'Excited', 'Friendly', 'Serious', 'Soothing', 'Broadcaster'
HEYGEN_EMOTION = "Friendly"  # Leave as None to use default voice emotion

# Optional parameters
AVATAR_STYLE = "normal"
BACKGROUND_COLOR = "#008000"  # Green background
OUTPUT_FILENAME = "heygen_voice_output.mp4"

# B-roll parameters (optional)
USE_BROLL = False  # Set to True to enable B-roll
BROLL_COUNT = 3
BROLL_DURATION = 5.0
ENGLISH_TEXT = None  # Provide English text when using non-English content
```

Then run:

```bash
python pipeline_heygen_voice.py
```

#### Using ElevenLabs

Edit `pipeline_elevenlabs.py` with your parameters:

```python
# Set your parameters here
TEXT = "Your text to convert to video."
FRONT_MODEL_ID = "Vernon_standing_lounge_front"
SIDE_MODEL_ID = "Vernon_standing_lounge_side"

# Optional parameters
AVATAR_STYLE = "normal"
BACKGROUND_COLOR = "#008000"  # Green background
OUTPUT_FILENAME = "elevenlabs_output.mp4"

# Audio splitting parameters (optional)
SILENCE_THRESHOLD = -50
MIN_SILENCE_LEN = 500
KEEP_SILENCE = 100

# B-roll parameters (optional)
USE_BROLL = False  # Set to True to enable B-roll
BROLL_COUNT = 3
BROLL_DURATION = 5.0
ENGLISH_TEXT = None  # Provide English text when using non-English content
```

Then run:

```bash
python pipeline_elevenlabs.py
```

### Advanced Usage

You can import and use the pipeline in your own Python code:

```python
import asyncio
from pipeline_runner import run_pipeline

async def main():
    result = await run_pipeline(
        text="Your text to convert to video",
        front_avatar_id="Vernon_standing_lounge_front",
        side_avatar_id="Vernon_standing_lounge_side",
        avatar_style="normal",
        background_color="#008000",
        output_filename="custom_video.mp4",
        
        # Voice settings - choose one approach:
        
        # Option 1: Use HeyGen's built-in text-to-speech
        use_heygen_voice=True,
        heygen_voice_id="b5db595bfd744dfc9a2087c5822cf29b",
        heygen_emotion="Friendly",  # Optional
        
        # Option 2: Use ElevenLabs
        # use_heygen_voice=False,
        # silence_threshold=-50,  # Optional audio splitting parameters
        # min_silence_len=500,
        # keep_silence=100,
        
        # B-roll settings (optional)
        use_broll=False,
        broll_count=3,
        broll_duration=5.0,
        broll_text_english=None,  # For non-English content
        broll_orientation="landscape",
        broll_video_size="medium"
    )
    
    if result["success"]:
        print(f"Video created: {result['final_video']}")
    else:
        print(f"Error: {result['error']}")

# Run the async function
asyncio.run(main())
```

## Voice Generation Options

The pipeline supports two different methods for generating speech:

### 1. ElevenLabs Mode

This mode:
- Uses ElevenLabs API for high-quality text-to-speech
- Splits audio at natural pauses using silence detection
- Uploads audio files to public URLs for HeyGen to access
- Creates videos using those audio URLs

**Pros:**
- Greater control over audio generation
- Can use any ElevenLabs voice

**Cons:**
- Requires an ElevenLabs API key
- Involves more steps (audio generation, silence splitting, uploading)

### 2. HeyGen Voice API Mode

This mode:
- Uses HeyGen's built-in text-to-speech capability
- Segments text into natural chunks
- Sends text directly to HeyGen along with a voice ID

**Pros:**
- Simpler pipeline with fewer steps
- No need for audio upload
- Only requires a HeyGen API key

**Cons:**
- Limited to voices available in HeyGen's system

To use HeyGen's voice API, set `use_heygen_voice=True` and provide your `heygen_voice_id`.

## How It Works

The pipeline follows a clear workflow based on the selected mode:

### ElevenLabs Mode:
1. **Text-to-Speech**: Converts your text to high-quality audio using ElevenLabs
2. **Audio Segmentation**: Splits the audio at natural pauses using silence detection
3. **Audio Clustering**: Groups audio segments into optimal clusters
4. **Audio Upload**: Uploads audio clusters to public URLs for HeyGen to access
5. **Video Generation**: Sends audio URLs to HeyGen API to generate video segments with alternating avatar poses
6. **Video Download**: Downloads completed video segments
7. **Video Merging**: Combines all segments into a seamless final video
8. **B-roll Processing** (optional): Adds relevant B-roll footage to the final video

### HeyGen Voice Mode:
1. **Text Segmentation**: Splits your text into chunks of approximately 9-11 seconds each
2. **Text Clustering**: Groups text segments into optimal clusters
3. **Video Generation**: Sends text directly to HeyGen API along with voice ID to generate video segments with alternating avatar poses
4. **Video Download**: Downloads completed video segments
5. **Video Merging**: Combines all segments into a seamless final video
6. **B-roll Processing** (optional): Adds relevant B-roll footage to the final video

This creates optimal pacing with scene changes every 9-11 seconds (3-4 switches in a 40-second video).

## B-Roll Post-Processing

The pipeline supports adding B-roll footage to the final videos for a more engaging and professional look. This feature uses the Pexels API to automatically find and insert relevant video clips based on keywords extracted from your text.

### Requirements for B-Roll

To use B-roll post-processing, you'll need:

1. **Pexels API Key**: Store this in your `.env` file as `PEXELS_API_KEY`
2. **Text Content in English**: For best results, provide an English version of your text when using non-English content

### Using B-Roll

Enable B-roll processing by setting `use_broll=True` in your parameters:

```python
# B-roll post-processing parameters
USE_BROLL = True  # Set to True to enable B-roll post-processing
BROLL_COUNT = 3  # Number of B-roll segments to insert
BROLL_DURATION = 5.0  # Duration of each B-roll segment in seconds

# If your main text is not in English, provide an English translation
ENGLISH_TEXT = "Your text in English for keyword extraction..."

# B-roll video parameters
BROLL_ORIENTATION = "landscape"  # Options: landscape, portrait, square
BROLL_VIDEO_SIZE = "medium"  # Options: large (4K), medium (Full HD), small (HD)
```

### How B-Roll Processing Works

The B-roll post-processing follows these steps:

1. **Keyword Extraction**: Analyzes your text to identify key themes and subjects
2. **B-roll Search**: Uses the Pexels API to find relevant video clips based on extracted keywords
3. **Intelligent Placement**: Determines optimal positions in your video to insert B-roll footage
4. **Video Processing**: Inserts B-roll while maintaining audio continuity from the original video
5. **Final Composition**: Creates a professional-looking video that alternates between the avatar and relevant B-roll footage

### Tips for Better B-Roll Results

- Provide descriptive, concrete text that mentions specific objects, actions, or scenes
- When using non-English text, always provide an English translation via `broll_text_english`
- Adjust `broll_count` and `broll_duration` based on your video length (3-4 B-roll segments work well for a 40-second video)
- Set `broll_orientation` to match your avatar video's aspect ratio

## Troubleshooting

- Check the log output for detailed information about each stage
- Ensure your API keys are correct and have sufficient credits
- Verify your internet connection is stable for API calls
- If using ElevenLabs mode, adjust silence detection parameters if audio segmentation isn't working as expected
- If using HeyGen Voice mode, ensure your HeyGen voice ID is correct
- For B-roll issues, verify your Pexels API key and check if your keywords are specific enough to generate relevant results

## License

[MIT License](LICENSE)