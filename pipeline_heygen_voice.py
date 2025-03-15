#!/usr/bin/env python3
# pipeline_heygen_voice.py
"""
Entry point for the text-to-video pipeline using HeyGen's voice API.
"""

import asyncio
from pipeline_runner import run_pipeline

if __name__ == "__main__":
    # Set your parameters here
    TEXT = """
        Educația românească intră într-o nouă eră: Ministerul Educației devine Ministerul Educației și Cercetării!
        Hotărârea nr. 187 din 27 februarie 2025 deschide drumul spre inovație și conectează cercetarea la sistemul de învățământ.
        Elevii, studenții, profesorii și chiar antreprenorii au acum șansa să profite de un viitor mai dinamic și orientat spre tehnologie.
        Citește articolul complet și vezi cum schimbă această decizie piața muncii și viitorul României.
        Tu crezi că e începutul unei revoluții în educație?
        Platforma „ai aflat” este asistentul AI pentru legile din România, unde poți afla orice despre legi!
    """
    
    # Required parameters: avatar IDs for front and side poses
    FRONT_MODEL_ID = "b5cd56e33c3e4c90bf62ac67a3f0572b" # woman_prim_plan_gesturi_front
    SIDE_MODEL_ID = "db32dd569c52494b91d5875b07356b59" # woman_plan_mediu_gesturi_side

    # Required: HeyGen voice ID
    HEYGEN_VOICE_ID = "b5db595bfd744dfc9a2087c5822cf29b"
    
    # Optional: HeyGen voice emotion
    # Options: 'Excited', 'Friendly', 'Serious', 'Soothing', 'Broadcaster'
    # Leave as None to use default voice emotion
    HEYGEN_EMOTION = "Friendly"
    
    # Optional styling parameters
    AVATAR_STYLE = "normal"  # Options: normal, happy, sad, etc.
    BACKGROUND_COLOR = "#008000"  # Green background
    WIDTH = 1280  # Video width in pixels
    HEIGHT = 720  # Video height in pixels
    OUTPUT_FILENAME = "heygen_voice_output.mp4"
    
    CLUSTER_SIZE = 1  # Use 1 to keep chunks separate (9-11 seconds each)
    
    # Note: Text will be segmented into ~9-11 second chunks
    # This creates optimal pacing with 3-4 scene switches in a 40-second video

    # Run the pipeline with HeyGen voice mode
    result = asyncio.run(
        run_pipeline(
            text=TEXT,
            front_avatar_id=FRONT_MODEL_ID,
            side_avatar_id=SIDE_MODEL_ID,
            heygen_voice_id=HEYGEN_VOICE_ID,
            heygen_emotion=HEYGEN_EMOTION,
            avatar_style=AVATAR_STYLE,
            background_color=BACKGROUND_COLOR,
            width=WIDTH,
            height=HEIGHT,
            output_filename=OUTPUT_FILENAME,
            cluster_size=CLUSTER_SIZE,
            use_heygen_voice=True  # Explicitly using HeyGen voice mode
        )
    )
    
    # Print result summary
    print("\nPipeline result:")
    if result["success"]:
        print(f"✅ Success! Final video created: {result['final_video']}")
        print(f"Mode: HeyGen Voice")
        print(f"Total segments: {result['total_clusters']}")
        
        # Show avatar distribution
        if "avatar_distribution" in result:
            print("\nAvatar distribution:")
            for avatar_id, count in result["avatar_distribution"].items():
                avatar_type = "FRONT" if avatar_id == FRONT_MODEL_ID else "SIDE"
                print(f"  - {avatar_type} avatar: {count} segments")
    else:
        print(f"❌ Error: {result['error']}")