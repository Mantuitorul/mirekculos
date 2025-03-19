#!/usr/bin/env python3
"""
Query enhancer for B-roll search.
Uses a small/cheap OpenAI model to generate better search queries.
"""

import os
import logging
import openai
from typing import List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class QueryEnhancer:
    """Enhances search queries for B-roll footage using a small OpenAI model"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo-instruct"):
        """
        Initialize the query enhancer.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Small OpenAI model to use (defaults to gpt-3.5-turbo-instruct, which is cheap)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Query enhancement will be disabled.")
        
        self.model = model
        self.client = None
        if self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
    
    def enhance_keywords(self, keywords: List[str], max_enhanced: int = 5) -> List[str]:
        """
        Enhance keywords for better B-roll search results.
        
        Args:
            keywords: Original keywords extracted from content
            max_enhanced: Maximum number of enhanced keywords to return
            
        Returns:
            Enhanced keywords for B-roll search
        """
        if not self.client or not keywords:
            return keywords
            
        try:
            logger.info(f"Enhancing {len(keywords)} keywords for better B-roll search")
            
            # Construct a prompt for the model
            prompt = (
                "I'm searching for B-roll footage for a video. "
                f"Based on these extracted keywords: {', '.join(keywords)}, "
                "suggest better, more specific search terms that would yield good B-roll results. "
                "Focus on visual terms that would translate well to video. "
                "Return only the search terms, one per line, no numbering or explanation. "
                f"Provide at most {max_enhanced} terms."
            )
            
            # Query the model
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            
            # Process the response
            enhanced_text = response.choices[0].text.strip()
            
            # Split by newlines and clean up
            enhanced_keywords = [k.strip() for k in enhanced_text.split('\n') if k.strip()]
            
            # Limit to max_enhanced
            enhanced_keywords = enhanced_keywords[:max_enhanced]
            
            logger.info(f"Enhanced keywords: {enhanced_keywords}")
            return enhanced_keywords
            
        except Exception as e:
            logger.error(f"Error enhancing keywords: {str(e)}")
            # Fallback to original keywords if enhancement fails
            return keywords 