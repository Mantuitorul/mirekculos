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
        Distracția este una dintre cele mai mari plăceri ale vieții! 
        Indiferent dacă joci sportul tău preferat, petreci timp cu 
        prietenii și familia sau doar te relaxezi cu o carte bună, 
        distracția poate lua multe forme. Unii oameni găsesc bucurie 
        în activități creative, precum pictura sau muzica, în timp 
        ce alții preferă aventurile în aer liber, cum ar fi drumețiile 
        sau înotul. Cheia este să descoperi ce te face să zâmbești 
        și să te bucuri pe deplin de acele momente. Nu uita să râzi, 
        să dansezi și să te joci – aceste gesturi simple pot însenina 
        chiar și cele mai întunecate zile. Viața e prea scurtă pentru 
        a fi mereu serioși, așa că asigură-te că îți faci loc pentru 
        bucurie și joacă în rutina ta zilnică.
    """
    
    # Required parameters: avatar IDs for front and side poses
    FRONT_MODEL_ID  = "b5cd56e33c3e4c90bf62ac67a3f0572b"    # woman_prim_plan_gesturi_front
    SIDE_MODEL_ID   = "db32dd569c52494b91d5875b07356b59"    # woman_plan_mediu_gesturi_side

    # Required: HeyGen voice ID
    HEYGEN_VOICE_ID = "1d0fc346c5be4943acc7e460e9b27344"
    
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
    
    # Text clustering parameter
    CLUSTER_SIZE = 3  # Number of text chunks per cluster
    
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