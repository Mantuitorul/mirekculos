#!/usr/bin/env python3
"""
Text processing module for the pipeline.
Handles text segmentation, clustering, and ChatGPT integration.
"""

import re
import json
import logging
import time
import nltk
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

# Constants for estimating speech duration
CHARS_PER_SECOND = 15
TARGET_MIN_SECONDS = 9
TARGET_MAX_SECONDS = 11

# Download NLTK resources if needed
try:
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
        self.api_key = api_key
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
        
        "Dorim ca reelul să aibă segmente cu cadre de 9-11 secunde, așadar trebuie să împărțim scenariul în cadrele exacte pe care trebuie să le filmăm. "
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
            
            return segments
            
        except Exception as e:
            logger.error(f"Error structuring article: {str(e)}")
            raise

def split_text_into_chunks(text: str) -> List[str]:
    """
    Split text into natural chunks based on sentence boundaries,
    targeting chunks that would be approximately 9-11 seconds when spoken.
    
    Args:
        text: The input text to split
        
    Returns:
        List of text chunks optimized for the specified duration range
    """
    # Estimate characters per chunk based on target duration
    min_chars_per_chunk = TARGET_MIN_SECONDS * CHARS_PER_SECOND
    max_chars_per_chunk = TARGET_MAX_SECONDS * CHARS_PER_SECOND
    
    logger.info(f"Targeting chunks between {TARGET_MIN_SECONDS}-{TARGET_MAX_SECONDS} seconds")
    logger.info(f"Character range: {min_chars_per_chunk}-{max_chars_per_chunk} chars")
    
    # Clean up text (remove multiple spaces, etc.)
    text = ' '.join(text.split())
    
    # Split on sentence boundaries (., !, ?)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Make sure sentences end with punctuation
    for i in range(len(sentences)):
        if i < len(sentences) - 1 and not sentences[i].rstrip().endswith(('.', '!', '?')):
            sentences[i] = sentences[i] + '.'
    
    logger.info(f"Split text into {len(sentences)} sentences")
    
    # Build chunks
    chunks = []
    current_chunk = ""
    current_chunk_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed the max chunk size, start a new chunk
        if current_chunk_size + sentence_size > max_chars_per_chunk and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_chunk_size = 0
        
        # For regular sentences, add to current chunk
        new_chunk_size = current_chunk_size + sentence_size
        if current_chunk:
            new_chunk_size += 1  # Account for space
        
        # Check if adding this sentence would push us beyond max_chars_per_chunk
        if new_chunk_size > max_chars_per_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_chunk_size = sentence_size
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_chunk_size = new_chunk_size
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Final validation
    for i, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        duration = chunk_size / CHARS_PER_SECOND
        
        if duration < TARGET_MIN_SECONDS:
            logger.warning(f"Chunk {i+1} is shorter than target minimum: {duration:.1f}s < {TARGET_MIN_SECONDS}s")
        elif duration > TARGET_MAX_SECONDS:
            logger.warning(f"Chunk {i+1} is longer than target maximum: {duration:.1f}s > {TARGET_MAX_SECONDS}s")
    
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks

def cluster_text_chunks(chunks: List[str], group_size: int = 1) -> List[Dict[str, Any]]:
    """
    Clusters a list of text chunks into groups of maximum `group_size` chunks.
    Each group is concatenated into a single string with appropriate spacing.
    
    Args:
        chunks: List of text chunks
        group_size: Maximum number of chunks in each cluster (use 1 for 9-11 second chunks)
        
    Returns:
        A list of dictionaries with cluster text and metadata including original positions
    """
    logger.info(f"Clustering {len(chunks)} text chunks with max group size {group_size}")
    
    if not chunks:
        logger.warning("No chunks to cluster, returning empty list")
        return []
    
    clusters = []
    for i in range(0, len(chunks), group_size):
        # Get the current group of chunks
        group = chunks[i:i+group_size]
        
        if group:
            # Record the original positions and join text with spaces
            start_pos = i
            end_pos = i + len(group) - 1
            cluster_text = " ".join(group)
            cluster_duration = len(cluster_text) / CHARS_PER_SECOND
            
            # Create cluster with metadata
            cluster = {
                "text": cluster_text,
                "start_chunk": start_pos,
                "end_chunk": end_pos,
                "num_chunks": len(group),
                "char_length": len(cluster_text),
                "duration": cluster_duration,
                "original_order": len(clusters)  # Keep track of original order
            }
            
            clusters.append(cluster)
            logger.info(f"Created cluster {len(clusters)} with chunks {start_pos}-{end_pos}, "
                       f"length: {len(cluster_text)} chars, duration: {cluster_duration:.1f}s")
    
    logger.info(f"Created {len(clusters)} text clusters in total")
    return clusters

def get_chunk_durations(chunks: List[str]) -> List[float]:
    """
    Estimate the duration of each text chunk in seconds.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of estimated durations in seconds
    """
    return [len(chunk) / CHARS_PER_SECOND for chunk in chunks]

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
            except Exception:
                # Fall back to English if the language isn't available
                logger.warning(f"Stopwords not available for {language}, using English")
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
        
        # Use appropriate extraction method
        if NLTK_AVAILABLE:
            try:
                return self._extract_with_nltk(text, max_keywords)
            except Exception as e:
                logger.warning(f"NLTK extraction failed: {str(e)}. Using simple method.")
                return self._extract_simple(text, max_keywords)
        else:
            return self._extract_simple(text, max_keywords)
    
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
        logger.info(f"Extracted keywords with NLTK: {keywords}")
        return keywords
    
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
        
    def determine_broll_points(
        self, 
        video_files: List[str], 
        num_points: int = 2,
        min_spacing: float = 10.0,
        edge_buffer: float = 5.0,
        fixed_interval: float = 7.0
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
        # This function is retained from the original code but condensed
        # Implementation would involve iterating through video files,
        # calculating positions based on the fixed interval, and returning
        # a list of insertion points with timing information
        from moviepy.video.io.VideoFileClip import VideoFileClip
        
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
        return points