#!/bin/bash
# Example cURL commands for using the reel generation API

# API base URL 
API_URL="http://localhost:8000"

# 1. Generate a new reel
# This will start a new reel generation job with the specified parameters
echo "Generating a new reel..."
response=$(curl -s -X POST "$API_URL/generate-reel" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Educația românească intră într-o nouă eră: Ministerul Educației devine Ministerul Educației și Cercetării!",
    "voice_id_1": "a426f8a763824ceaad3a2eb29c68e121",
    "look_id_1": "Raul_sitting_sofa_front_close",
    "look_id_2": "Raul_sitting_sofa_side_close",
    "emotion": "Friendly",
    "remove_silence": true,
    "background_color": "#008000"
  }')

# Extract the process ID from the response
process_id=$(echo $response | grep -o '"process_id":"[^"]*' | cut -d'"' -f4)

if [ -n "$process_id" ]; then
  echo "Process started with ID: $process_id"
  
  # 2. Check process status
  echo -e "\nChecking process status..."
  curl -s -X GET "$API_URL/process/$process_id" | jq .
  
  echo -e "\nNote: The process will take some time to complete. You can check status again with:"
  echo "curl -X GET $API_URL/process/$process_id"
  
  echo -e "\nTo download the video once completed:"
  echo "curl -X GET $API_URL/download/$process_id -o reel_$process_id.mp4"
else
  echo "Failed to start process."
  echo "Response: $response"
fi

# Other useful commands:

echo -e "\n--- Other Useful Commands ---"

echo -e "\n3. List all processes:"
echo "curl -X GET $API_URL/processes"

echo -e "\n4. Check API health:"
echo "curl -X GET $API_URL/health"

echo -e "\n5. Delete a process:"
echo "curl -X DELETE $API_URL/process/{process_id}"

echo -e "\n6. Download original video without silence removal:"
echo "curl -X GET \"$API_URL/download/{process_id}?no_silence=false\" -o reel_with_silence.mp4"