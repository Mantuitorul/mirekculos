#!/usr/bin/env python3
# post_processing/content_analyzer.py
"""
Content analysis for B-roll insertion.
Extracts keywords and determines optimal insertion points.
"""

import re
import logging
import nltk
from typing import List, Dict, Any, Tuple
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip  # noqa: E402

# Configure logging
logger = logging.getLogger(__name__)

# Pre-download NLTK resources
try:
    # The correct resources are 'punkt' and 'stopwords'
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except Exception as e:
    logger.warning(f"NLTK resource download failed: {str(e)}")
    NLTK_AVAILABLE = False

# Import NLTK modules with fallback
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError:
    logger.warning("NLTK corpus modules not available, using fallback methods")
    NLTK_AVAILABLE = False

class ContentAnalyzer:
    """Analyzes content to extract keywords and determine B-roll points"""
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the content analyzer.
        
        Args:
            language: Language for stopwords ('english', 'romanian', etc.)
        """
        self.language = language
        
        # Create stopwords set based on availability
        self.stop_words = set()
        if NLTK_AVAILABLE:
            try:
                # Try to get stopwords for the specified language
                self.stop_words = set(stopwords.words(language))
                logger.info(f"Using NLTK stopwords for {language}")
            except Exception as e:
                # Fall back to English if the language isn't available
                logger.warning(f"Stopwords not available for {language}, using English: {str(e)}")
                try:
                    self.stop_words = set(stopwords.words('english'))
                except:
                    # Basic stopword set if NLTK fails completely
                    self.stop_words = self._get_basic_stopwords()
        else:
            # Use basic stopwords list if NLTK isn't available
            self.stop_words = self._get_basic_stopwords()
            
        # Add common words that aren't useful for B-roll searches
        self.stop_words.update(['would', 'could', 'should', 'like', 'also', 'well'])
    
    def _get_basic_stopwords(self) -> set:
        """Provides basic stopwords when NLTK isn't available"""
        return set(['the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'of',
                   'at', 'to', 'for', 'with', 'by', 'about', 'against', 'between',
                   'into', 'through', 'during', 'before', 'after', 'above', 'below',
                   'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                   'again', 'further', 'then', 'once', 'here', 'there', 'when',
                   'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                   'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                   'only', 'own', 'same', 'so', 'than', 'too', 'very', 'which',
                   'would', 'could', 'should', 'like', 'also', 'well',
                   # Romanian stopwords
                   'și', 'în', 'a', 'ca', 'pe', 'este', 'de', 'la', 'cu', 'pentru',
                   'ce', 'nu', 'o', 'mai', 'sa', 'se', 'din', 'sau', 'sunt', 'care',
                   'dar', 'ai', 'ale'])
        
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract the most relevant keywords from text for B-roll searching.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        logger.info(f"Extracting keywords from text ({len(text)} chars)")
        
        if not text.strip():
            logger.warning("Empty text provided for keyword extraction")
            return []
            
        # Detect if text is not English and adjust for that
        is_english = self._is_text_english(text)
        if not is_english and self.language == 'english':
            logger.info("Text appears to be non-English, adjusting extraction method")
        
        # Use appropriate extraction method
        if NLTK_AVAILABLE:
            try:
                return self._extract_with_nltk(text, max_keywords)
            except Exception as e:
                logger.warning(f"NLTK extraction failed: {str(e)}. Using simple method.")
                return self._extract_simple(text, max_keywords)
        else:
            return self._extract_simple(text, max_keywords)
    
    def _is_text_english(self, text: str) -> bool:
        """
        Determine if text is likely English based on common character frequency.
        This is a simple heuristic, not foolproof.
        """
        # Common English letters
        english_chars = set('etaoinsrhdlucmfywgpbvkjxqz')
        # Common letters in Romanian and other Latin alphabets that use diacritics
        non_english_chars = set('ăâîșțéáóúíäëïöüàèìòùç')
        
        text = text.lower()
        english_count = sum(1 for c in text if c in english_chars)
        non_english_count = sum(1 for c in text if c in non_english_chars)
        
        # If significant non-English characters present, assume non-English
        return non_english_count < english_count * 0.05
    
    def _extract_with_nltk(self, text: str, max_keywords: int) -> List[str]:
        """Extract keywords using NLTK's tokenization and analysis"""
        # Tokenize text and remove punctuation
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]
        
        # Remove stop words
        filtered_words = [word for word in words if word not in self.stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            if len(word) > 3:  # Ignore very short words
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top keywords
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        # Add two-word phrases for better search results
        phrases = self._extract_phrases(text, keywords)
        
        # Combine individual keywords and phrases
        combined_keywords = []
        for phrase in phrases[:2]:  # Take top 2 phrases
            if phrase not in combined_keywords:
                combined_keywords.append(phrase)
                
        # Add remaining space with individual keywords
        for keyword in keywords:
            if len(combined_keywords) >= max_keywords:
                break
            if keyword not in combined_keywords:
                combined_keywords.append(keyword)
                
        logger.info(f"Extracted keywords with NLTK: {combined_keywords}")
        return combined_keywords
    
    def _extract_simple(self, text: str, max_keywords: int) -> List[str]:
        """Extract keywords using simple string operations (fallback method)"""
        # Clean text and split into words
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        
        # Remove stop words
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 3]
        
        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top keywords
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        logger.info(f"Extracted keywords with simple method: {keywords}")
        return keywords
        
    def _extract_phrases(self, text: str, keywords: List[str]) -> List[str]:
        """
        Extract meaningful two-word phrases containing important keywords.
        
        Args:
            text: Text to analyze
            keywords: List of important keywords
            
        Returns:
            List of phrases
        """
        try:
            sentences = sent_tokenize(text.lower())
        except Exception as e:
            # Fall back to simple sentence splitting
            logger.warning(f"Sentence tokenization failed: {str(e)}. Using simple splitting.")
            sentences = re.split(r'[.!?]+', text.lower())
            
        phrases = []
        
        for sentence in sentences:
            # Clean and tokenize
            clean_sent = re.sub(r'[^\w\s]', '', sentence)
            words = clean_sent.split()
            
            # Find two-word phrases containing keywords
            for i in range(len(words) - 1):
                if words[i] in keywords or words[i + 1] in keywords:
                    if words[i] not in self.stop_words and words[i + 1] not in self.stop_words:
                        phrase = f"{words[i]} {words[i + 1]}"
                        if len(phrase) > 5:  # Avoid very short phrases
                            phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_freq = {}
        for phrase in phrases:
            phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
            
        # Sort by frequency
        sorted_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [phrase for phrase, freq in sorted_phrases]
        
    def determine_broll_points(
        self, 
        video_files: List[str], 
        num_points: int = 2,
        min_spacing: float = 10.0,
        edge_buffer: float = 5.0,
        fixed_interval: float = 7.0  # New parameter with default 7 seconds
    ) -> List[Dict[str, Any]]:
        """
        Determine optimal points for B-roll insertion using fixed intervals.
        
        Args:
            video_files: List of video segment files
            num_points: Maximum number of B-roll insertion points
            min_spacing: Minimum spacing between B-roll points (not used for fixed interval)
            edge_buffer: Minimum distance from start/end of complete video
            fixed_interval: Fixed interval in seconds between B-roll clips
            
        Returns:
            List of B-roll insertion points with timing information
        """
        logger.info(f"Determining B-roll points at {fixed_interval}s intervals for {len(video_files)} video segments")
        
        # Get total duration and build timeline
        timeline = []
        total_duration = 0
        
        for idx, video_path in enumerate(video_files):
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                timeline.append({
                    "file_index": idx,
                    "file_path": video_path,
                    "start_time": total_duration,
                    "end_time": total_duration + duration,
                    "duration": duration
                })
                total_duration += duration
                clip.close()  # Make sure to close the clip to release resources
            except Exception as e:
                logger.error(f"Error analyzing video file {video_path}: {str(e)}")
                # Continue with other files if one fails
                continue
        
        if not timeline:
            logger.error("No valid video files could be analyzed")
            return []
            
        logger.info(f"Total video duration: {total_duration:.2f}s")
        
        # Never insert at the very beginning or end
        valid_start = edge_buffer
        valid_end = total_duration - edge_buffer
        valid_duration = valid_end - valid_start
        
        # Calculate how many points we can fit with the fixed interval
        max_possible_points = max(1, int(valid_duration / fixed_interval))
        
        # Respect the num_points limit if provided
        actual_points = min(max_possible_points, num_points) if num_points > 0 else max_possible_points
        
        points = []
        # Place points at fixed intervals
        for i in range(1, actual_points + 1):
            point_time = valid_start + (fixed_interval * i)
            
            # Ensure we don't exceed the valid end time
            if point_time >= valid_end:
                break
            
            # Find which segment contains this point
            for segment in timeline:
                if segment["start_time"] <= point_time < segment["end_time"]:
                    # Calculate time within this specific file
                    file_relative_time = point_time - segment["start_time"]
                    
                    point = {
                        "global_time": point_time,
                        "file_index": segment["file_index"],
                        "file_path": segment["file_path"],
                        "file_time": file_relative_time
                    }
                    points.append(point)
                    break
        
        logger.info(f"Determined {len(points)} B-roll insertion points at {fixed_interval}s intervals")
        for p in points:
            logger.info(f"B-roll point at {p['global_time']:.2f}s (file {p['file_index']}, time {p['file_time']:.2f}s)")
            
        return points