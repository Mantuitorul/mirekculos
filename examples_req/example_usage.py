#!/usr/bin/env python3
"""
Example of how to use the reel generation API.
"""

import requests
import time
import json
import sys

# API endpoint (adjust if needed)
API_URL = "http://localhost:8000"

def generate_reel():
    """
    Example of generating a reel using the API.
    """
    # Example parameters
    request_data = {
        "text": "Educația românească intră într-o nouă eră: Ministerul Educației devine Ministerul Educației și Cercetării! Elevii, studenții, profesorii și chiar antreprenorii au acum șansa să profite de un viitor mai dinamic și orientat spre tehnologie.",
        "voice_id_1": "a426f8a763824ceaad3a2eb29c68e121",  # Example voice ID
        "look_id_1": "Raul_sitting_sofa_front_close",  # Front avatar
        "look_id_2": "Raul_sitting_sofa_side_close",   # Side avatar
        "emotion": "Friendly",
        "remove_silence": True,
        "silence_threshold": -34,  # Lower value = more sensitive
        "min_silence_duration": 0.3,
        "keep_silence_ratio": 0.2,
        "background_color": "#008000",  # Green background
        "debug": True
    }
    
    # Send request to generate reel
    print("Sending request to generate reel...")
    response = requests.post(f"{API_URL}/generate-reel", json=request_data)
    
    if response.status_code == 202:
        process_data = response.json()
        process_id = process_data["process_id"]
        print(f"Process started with ID: {process_id}")
        
        # Poll for process status
        print("Waiting for process to complete...")
        while True:
            status_response = requests.get(f"{API_URL}/process/{process_id}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']}")
                
                if status_data["status"] == "completed":
                    # Process completed successfully
                    print("\nProcess completed successfully!")
                    print(f"Result: {json.dumps(status_data['result'], indent=2)}")
                    
                    # Download the video
                    print("\nDownloading video...")
                    download_url = f"{API_URL}/download/{process_id}?no_silence=true"
                    download_response = requests.get(download_url)
                    
                    if download_response.status_code == 200:
                        # Save the video
                        output_filename = f"downloaded_reel_{process_id}.mp4"
                        with open(output_filename, "wb") as f:
                            f.write(download_response.content)
                        print(f"Video saved to: {output_filename}")
                    else:
                        print(f"Failed to download video: {download_response.text}")
                    
                    break
                elif status_data["status"] == "failed":
                    # Process failed
                    print(f"Process failed: {status_data.get('error', 'Unknown error')}")
                    break
                
                # Wait before polling again
                time.sleep(5)
            else:
                print(f"Error checking status: {status_response.text}")
                break
    else:
        print(f"Failed to start process: {response.text}")

def list_all_processes():
    """List all processes in the system."""
    response = requests.get(f"{API_URL}/processes")
    
    if response.status_code == 200:
        processes = response.json()
        print(f"Found {len(processes)} processes:")
        for process in processes:
            print(f"- Process {process['process_id']}: {process['status']}")
    else:
        print(f"Failed to list processes: {response.text}")

def check_health():
    """Check API health status."""
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        health_data = response.json()
        print("API Health Status:")
        print(f"- Status: {health_data['status']}")
        print("- API Keys Available:")
        for key, available in health_data.get('api_keys', {}).items():
            print(f"  - {key}: {'✅' if available else '❌'}")
        print(f"- Active Processes: {health_data.get('active_processes', 0)}")
    else:
        print(f"Failed to check health: {response.text}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "generate":
            generate_reel()
        elif command == "list":
            list_all_processes()
        elif command == "health":
            check_health()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: generate, list, health")
    else:
        # Default to health check
        check_health()
        
        # Ask user what to do
        print("\nWhat would you like to do?")
        print("1. Generate a new reel")
        print("2. List all processes")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            generate_reel()
        elif choice == "2":
            list_all_processes()
        else:
            print("Exiting...")