#!/usr/bin/env python3
# text/chatgpt_integration.py
"""
ChatGPT integration for structuring article text into video segments.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pathlib import Path
import time

# Configure logging
logger = logging.getLogger(__name__)

class VideoStructurer:
    """Structures article text into video segments using ChatGPT."""
    
    def __init__(self, api_key: Optional[str] = None, debug_mode: bool = False, debug_dir: Optional[Path] = None):
        """
        Initialize the VideoStructurer.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            debug_mode: If True, save the generated segments to a file for inspection
            debug_dir: Directory to save debug files (defaults to current directory)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("No OpenAI API key provided. Set OPENAI_API_KEY in .env file or pass with api_key parameter.")
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir or Path(".")
        
        if self.debug_mode:
            self.debug_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Debug mode enabled. Debug files will be saved to: {self.debug_dir}")
    
    def structure_article(self, article_text: str) -> List[Dict[str, Any]]:
        """
        Structure the article text into video segments using ChatGPT.
        
        Args:
            article_text: The article text to structure
            
        Returns:
            List of dictionaries with segment_text, segment_shot, and instructions
        """
        logger.info("Using ChatGPT to structure article into video segments")
        
        prompt = (
        "Avem un articol despre o lege din România. Articolul oferă detalii despre ce înseamnă legea și care sunt efectele acesteia. "
        "Dorim să creăm un reel video pentru Social Media bazat pe acest articol pentru a ne extinde audiența. "
        "Ne dorim ca reelul să fie informativ și captivant, ușor de înțeles și plăcut de vizionat.\n\n"
        
        f"Am elaborat aceste note despre conținutul articolului:\n\n"
        f"\"\"\"{article_text}\"\"\"\n\n"
        
        "Transformă aceasta în scenariul pentru reel-ul nostru, astfel încât să-l putem filma. Vom avea un actor de voce care va citi transcriptul scenariului. "
        "Videoclipul va avea mai multe cadre cu diferite unghiuri:\n"
        "- cadru side: potrivit pentru informații generale\n"
        "- cadru front: potrivit pentru momentele când se spune ceva important\n"
        "- cadru broll: potrivit pentru a arăta imagini conexe conținutului\n\n"
        
        "Dorim ca reelul să aibă segmente cu cadre de 3-7 secunde, așadar trebuie să împărțim scenariul în cadrele exacte pe care trebuie să le filmăm. "
        "Unele vor fi cadre side, altele front, altele broll. Trebuie să alegi cum să împarți cel mai bine scenariul în aceste segmente și ce tip de cadru se potrivește cel mai bine. "
        "Acestea trebuie să se alterneze în mod dinamic: cadru front după cel side, intercalat cu broll, etc.\n\n"
        
        "Rescrie draftul în scenariul pentru reel-ul de pe rețelele sociale astfel încât să fie cât mai captivant. Alege cel mai bun tip de cadru pentru fiecare segment.\n\n"
        
        "Ghid:\n"
        "- Fiecare segment ar trebui să constea dintr-o propoziție scurtă\n"
        "- evită propozițiile lungi și complicate.\n"
        "- folosește cadre largi pentru propoziții explicative și close-up-uri pentru afirmații de impact la care utilizatorul trebuie să acorde atenție\n"
        "- folosește cadre broll pentru segmentele care sunt vizual descriptive (pe cât posibil)\n\n"
        
        "Răspunde cu o listă de obiecte JSON cu segmentele rescrise, fiecare având structura:\n"
        "- segment_text: ceea ce spune naratorul în acel segment\n"
        "- segment_shot: front/side/broll\n"
        "- instructions: alte instrucțiuni pentru filmarea cadrului sau cum ar trebui să arate broll-ul"
    )
        
        try:
            # Query the model
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            # Process the response
            response_content = response.choices[0].message.content.strip()
            
            # Extract JSON list from the response
            try:
                # Try to parse the whole response as JSON
                segments = json.loads(response_content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON list using string manipulation
                start_idx = response_content.find('[')
                end_idx = response_content.rfind(']') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_content = response_content[start_idx:end_idx]
                    segments = json.loads(json_content)
                else:
                    raise ValueError(f"Could not extract JSON from response: {response_content}")
            
            logger.info(f"Created {len(segments)} video segments")
            
            # Save segments to file in debug mode
            if self.debug_mode:
                timestamp = int(time.time())
                debug_file = self.debug_dir / f"segments_{timestamp}.json"
                
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved segments to {debug_file}")
                
                # Also save the raw response for reference
                response_file = self.debug_dir / f"response_{timestamp}.txt"
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(response_content)
                
                logger.info(f"Saved raw response to {response_file}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error structuring article: {str(e)}")
            raise 