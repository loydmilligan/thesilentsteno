#!/usr/bin/env python3
"""
Topic Identifier Module

Key topic and theme identification system with clustering, importance scoring,
and trend analysis for meeting transcripts.

Author: Claude AI Assistant
Date: 2024-07-15
Version: 1.0
"""

import os
import sys
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
from collections import Counter, defaultdict
import math

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Warning: sklearn not available: {e}")
    print("Install with: pip install scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicType(Enum):
    """Types of topics"""
    MAIN_TOPIC = "main_topic"
    SUBTOPIC = "subtopic"
    DISCUSSION_POINT = "discussion_point"
    DECISION_POINT = "decision_point"
    CONCERN = "concern"
    OPPORTUNITY = "opportunity"


class ImportanceLevel(Enum):
    """Topic importance levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Topic:
    """Represents an identified topic"""
    
    # Core information
    id: str
    name: str
    description: str
    
    # Classification
    topic_type: TopicType = TopicType.MAIN_TOPIC
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    
    # Metrics
    relevance_score: float = 0.0
    time_spent: float = 0.0  # Percentage of meeting time
    mention_count: int = 0
    
    # Content
    keywords: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    
    # Participants
    participants: List[str] = field(default_factory=list)
    main_contributors: List[str] = field(default_factory=list)
    
    # Context
    context_segments: List[str] = field(default_factory=list)
    decisions_made: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "topic_type": self.topic_type.value,
            "importance": self.importance.value,
            "relevance_score": self.relevance_score,
            "time_spent": self.time_spent,
            "mention_count": self.mention_count,
            "keywords": self.keywords,
            "key_phrases": self.key_phrases,
            "related_topics": self.related_topics,
            "participants": self.participants,
            "main_contributors": self.main_contributors,
            "context_segments": self.context_segments,
            "decisions_made": self.decisions_made,
            "action_items": self.action_items,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TopicConfig:
    """Configuration for topic identification"""
    
    # Analysis settings
    max_topics: int = 5
    min_relevance_score: float = 0.3
    min_mentions: int = 2
    
    # Clustering settings
    use_clustering: bool = True
    n_clusters: int = 5
    min_cluster_size: int = 3
    
    # Text processing
    min_phrase_length: int = 2
    max_phrase_length: int = 5
    stop_words: List[str] = field(default_factory=lambda: [
        "the", "is", "are", "was", "were", "been", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "must", "shall", "this", "that", "these", "those",
        "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "among", "under", "over"
    ])
    
    # Importance scoring
    importance_factors: Dict[str, float] = field(default_factory=lambda: {
        "mention_frequency": 0.3,
        "time_spent": 0.2,
        "participant_count": 0.2,
        "decision_weight": 0.2,
        "keyword_density": 0.1
    })
    
    # Context settings
    context_window: int = 200  # Characters around mentions
    include_speaker_analysis: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_topics": self.max_topics,
            "min_relevance_score": self.min_relevance_score,
            "min_mentions": self.min_mentions,
            "use_clustering": self.use_clustering,
            "n_clusters": self.n_clusters,
            "min_cluster_size": self.min_cluster_size,
            "min_phrase_length": self.min_phrase_length,
            "max_phrase_length": self.max_phrase_length,
            "stop_words": self.stop_words,
            "importance_factors": self.importance_factors,
            "context_window": self.context_window,
            "include_speaker_analysis": self.include_speaker_analysis
        }


@dataclass
class TopicResult:
    """Result from topic identification"""
    
    # Identified topics
    topics: List[Topic] = field(default_factory=list)
    
    # Analysis metadata
    processing_time: float = 0.0
    total_topics_found: int = 0
    clusters_formed: int = 0
    
    # Statistics
    keywords_extracted: int = 0
    phrases_identified: int = 0
    participants_analyzed: int = 0
    
    # Topic relationships
    topic_relationships: Dict[str, List[str]] = field(default_factory=dict)
    topic_timeline: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    success: bool = True
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "topics": [topic.to_dict() for topic in self.topics],
            "processing_time": self.processing_time,
            "total_topics_found": self.total_topics_found,
            "clusters_formed": self.clusters_formed,
            "keywords_extracted": self.keywords_extracted,
            "phrases_identified": self.phrases_identified,
            "participants_analyzed": self.participants_analyzed,
            "topic_relationships": self.topic_relationships,
            "topic_timeline": self.topic_timeline,
            "success": self.success,
            "error_message": self.error_message,
            "warnings": self.warnings
        }


class TopicIdentifier:
    """Main topic identification system"""
    
    def __init__(self, config: Optional[TopicConfig] = None):
        self.config = config or TopicConfig()
        self.vectorizer = None
        self.is_sklearn_available = self._check_sklearn()
        
        # Topic patterns
        self.topic_indicators = [
            r"(?i)(topic|subject|issue|matter|point|question|problem|challenge)",
            r"(?i)(discuss|talk about|address|cover|review|examine|consider)",
            r"(?i)(regarding|concerning|about|related to|in terms of)",
            r"(?i)(main|primary|key|important|critical|significant|major)"
        ]
        
        # Decision indicators
        self.decision_indicators = [
            r"(?i)(decide|decided|decision|conclusion|agreement|consensus)",
            r"(?i)(resolve|resolved|solution|approved|accepted|rejected)",
            r"(?i)(vote|voted|unanimous|majority|agreed|disagree)"
        ]
        
        logger.info(f"TopicIdentifier initialized (sklearn: {self.is_sklearn_available})")
        
    def _check_sklearn(self) -> bool:
        """Check if sklearn is available"""
        try:
            import sklearn
            return True
        except ImportError:
            return False
            
    def identify_topics(self, transcript: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> TopicResult:
        """Identify topics from meeting transcript"""
        start_time = datetime.now()
        
        try:
            result = TopicResult()
            
            # Preprocess transcript
            processed_text = self._preprocess_transcript(transcript)
            
            # Extract keywords and phrases
            keywords = self._extract_keywords(processed_text)
            phrases = self._extract_key_phrases(processed_text)
            
            result.keywords_extracted = len(keywords)
            result.phrases_identified = len(phrases)
            
            # Identify topic candidates
            topic_candidates = self._identify_topic_candidates(processed_text, keywords, phrases)
            
            # Cluster topics if sklearn available
            if self.is_sklearn_available and self.config.use_clustering:
                clustered_topics = self._cluster_topics(topic_candidates, processed_text)
                result.clusters_formed = len(set(topic.topic_type for topic in clustered_topics))
            else:
                clustered_topics = topic_candidates
                
            # Score and rank topics
            scored_topics = self._score_topics(clustered_topics, processed_text)
            
            # Filter and limit topics
            filtered_topics = self._filter_topics(scored_topics)
            
            # Analyze topic relationships
            result.topic_relationships = self._analyze_topic_relationships(filtered_topics)
            
            # Create topic timeline
            result.topic_timeline = self._create_topic_timeline(filtered_topics, transcript)
            
            # Analyze participants if speaker labels present
            if self.config.include_speaker_analysis and ":" in transcript:
                self._analyze_topic_participants(filtered_topics, transcript)
                result.participants_analyzed = len(set(
                    participant for topic in filtered_topics 
                    for participant in topic.participants
                ))
                
            # Finalize result
            result.topics = filtered_topics[:self.config.max_topics]
            result.total_topics_found = len(filtered_topics)
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = True
            
            logger.info(f"Identified {len(result.topics)} topics")
            return result
            
        except Exception as e:
            logger.error(f"Topic identification failed: {e}")
            
            result = TopicResult()
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = False
            result.error_message = str(e)
            
            return result
            
    def _preprocess_transcript(self, transcript: str) -> str:
        """Preprocess transcript for topic identification"""
        # Remove speaker labels for text analysis
        text = re.sub(r'^\w+:\s*', '', transcript, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase for processing
        return text.lower()
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text)
        
        # Filter out stop words
        keywords = [word for word in words 
                   if word not in self.config.stop_words and len(word) > 2]
        
        # Count frequency
        word_counts = Counter(keywords)
        
        # Return most frequent keywords
        return [word for word, count in word_counts.most_common(50) if count >= 2]
        
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        phrases = []
        
        # Extract n-grams
        words = text.split()
        for n in range(self.config.min_phrase_length, self.config.max_phrase_length + 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                
                # Filter out phrases with stop words
                if not any(stop_word in phrase for stop_word in self.config.stop_words):
                    phrases.append(phrase)
                    
        # Count frequency
        phrase_counts = Counter(phrases)
        
        # Return most frequent phrases
        return [phrase for phrase, count in phrase_counts.most_common(30) if count >= 2]
        
    def _identify_topic_candidates(self, text: str, keywords: List[str], 
                                 phrases: List[str]) -> List[Topic]:
        """Identify topic candidates from keywords and phrases"""
        candidates = []
        
        # Create topics from frequent keywords
        for keyword in keywords[:20]:  # Top 20 keywords
            if len(keyword) > 3:  # Skip very short words
                topic = Topic(
                    id=str(uuid.uuid4()),
                    name=keyword.capitalize(),
                    description=f"Topic related to {keyword}",
                    keywords=[keyword],
                    mention_count=text.count(keyword)
                )
                candidates.append(topic)
                
        # Create topics from key phrases
        for phrase in phrases[:15]:  # Top 15 phrases
            topic = Topic(
                id=str(uuid.uuid4()),
                name=phrase.title(),
                description=f"Discussion about {phrase}",
                key_phrases=[phrase],
                mention_count=text.count(phrase)
            )
            candidates.append(topic)
            
        return candidates
        
    def _cluster_topics(self, topics: List[Topic], text: str) -> List[Topic]:
        """Cluster similar topics together"""
        if not self.is_sklearn_available or len(topics) < 3:
            return topics
            
        try:
            # Prepare text for clustering
            topic_texts = []
            for topic in topics:
                topic_text = topic.name + " " + " ".join(topic.keywords) + " " + " ".join(topic.key_phrases)
                topic_texts.append(topic_text)
                
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(topic_texts)
            
            # Cluster
            n_clusters = min(self.config.n_clusters, len(topics))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group topics by cluster
            clustered_topics = []
            for cluster_id in range(n_clusters):
                cluster_topics = [topics[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_topics) >= self.config.min_cluster_size:
                    # Merge topics in cluster
                    merged_topic = self._merge_topics(cluster_topics)
                    clustered_topics.append(merged_topic)
                else:
                    # Keep individual topics
                    clustered_topics.extend(cluster_topics)
                    
            return clustered_topics
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return topics
            
    def _merge_topics(self, topics: List[Topic]) -> Topic:
        """Merge similar topics"""
        if len(topics) == 1:
            return topics[0]
            
        # Create merged topic
        merged = Topic(
            id=str(uuid.uuid4()),
            name=topics[0].name,  # Use first topic's name
            description=f"Merged topic from {len(topics)} similar topics",
            keywords=list(set(kw for topic in topics for kw in topic.keywords)),
            key_phrases=list(set(phrase for topic in topics for phrase in topic.key_phrases)),
            mention_count=sum(topic.mention_count for topic in topics)
        )
        
        return merged
        
    def _score_topics(self, topics: List[Topic], text: str) -> List[Topic]:
        """Score topics based on importance factors"""
        for topic in topics:
            # Calculate individual scores
            mention_score = self._calculate_mention_score(topic, text)
            time_score = self._calculate_time_score(topic, text)
            keyword_score = self._calculate_keyword_score(topic, text)
            decision_score = self._calculate_decision_score(topic, text)
            
            # Combine scores
            factors = self.config.importance_factors
            relevance_score = (
                factors["mention_frequency"] * mention_score +
                factors["time_spent"] * time_score +
                factors["keyword_density"] * keyword_score +
                factors["decision_weight"] * decision_score
            )
            
            topic.relevance_score = relevance_score
            topic.confidence = min(1.0, relevance_score)
            
            # Determine importance level
            if relevance_score > 0.8:
                topic.importance = ImportanceLevel.CRITICAL
            elif relevance_score > 0.6:
                topic.importance = ImportanceLevel.HIGH
            elif relevance_score > 0.4:
                topic.importance = ImportanceLevel.MEDIUM
            else:
                topic.importance = ImportanceLevel.LOW
                
        return topics
        
    def _calculate_mention_score(self, topic: Topic, text: str) -> float:
        """Calculate score based on mention frequency"""
        total_mentions = 0
        
        for keyword in topic.keywords:
            total_mentions += text.count(keyword)
            
        for phrase in topic.key_phrases:
            total_mentions += text.count(phrase)
            
        # Normalize by text length
        text_length = len(text.split())
        if text_length > 0:
            return min(1.0, total_mentions / (text_length / 100))
        else:
            return 0.0
            
    def _calculate_time_score(self, topic: Topic, text: str) -> float:
        """Calculate score based on time spent on topic"""
        # Approximate time spent by counting sentences containing topic
        sentences = text.split('.')
        topic_sentences = 0
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in topic.keywords):
                topic_sentences += 1
            if any(phrase in sentence for phrase in topic.key_phrases):
                topic_sentences += 1
                
        if len(sentences) > 0:
            time_ratio = topic_sentences / len(sentences)
            topic.time_spent = time_ratio * 100  # Convert to percentage
            return min(1.0, time_ratio * 2)  # Boost score
        else:
            return 0.0
            
    def _calculate_keyword_score(self, topic: Topic, text: str) -> float:
        """Calculate score based on keyword density"""
        if not topic.keywords:
            return 0.0
            
        keyword_density = sum(text.count(kw) for kw in topic.keywords) / len(text.split())
        return min(1.0, keyword_density * 100)
        
    def _calculate_decision_score(self, topic: Topic, text: str) -> float:
        """Calculate score based on decision-making context"""
        decision_score = 0.0
        
        # Look for decision indicators near topic mentions
        for keyword in topic.keywords:
            for match in re.finditer(re.escape(keyword), text):
                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 100)
                context = text[context_start:context_end]
                
                for pattern in self.decision_indicators:
                    if re.search(pattern, context):
                        decision_score += 0.2
                        
        return min(1.0, decision_score)
        
    def _filter_topics(self, topics: List[Topic]) -> List[Topic]:
        """Filter topics based on configuration"""
        filtered = []
        
        for topic in topics:
            if (topic.relevance_score >= self.config.min_relevance_score and
                topic.mention_count >= self.config.min_mentions):
                filtered.append(topic)
                
        # Sort by relevance score
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return filtered
        
    def _analyze_topic_relationships(self, topics: List[Topic]) -> Dict[str, List[str]]:
        """Analyze relationships between topics"""
        relationships = {}
        
        for topic in topics:
            related = []
            
            for other_topic in topics:
                if topic.id != other_topic.id:
                    # Check for keyword overlap
                    keyword_overlap = set(topic.keywords) & set(other_topic.keywords)
                    if len(keyword_overlap) > 0:
                        related.append(other_topic.id)
                        
            relationships[topic.id] = related
            
        return relationships
        
    def _create_topic_timeline(self, topics: List[Topic], transcript: str) -> List[Dict[str, Any]]:
        """Create timeline of topic mentions"""
        timeline = []
        
        # Simple timeline based on first mention
        for topic in topics:
            for keyword in topic.keywords:
                match = re.search(re.escape(keyword), transcript)
                if match:
                    timeline.append({
                        "topic_id": topic.id,
                        "topic_name": topic.name,
                        "position": match.start(),
                        "keyword": keyword
                    })
                    break
                    
        # Sort by position
        timeline.sort(key=lambda x: x["position"])
        
        return timeline
        
    def _analyze_topic_participants(self, topics: List[Topic], transcript: str):
        """Analyze which participants discussed which topics"""
        # Extract speaker segments
        speaker_segments = re.findall(r'(\w+):\s*([^:]+?)(?=\w+:|$)', transcript, re.DOTALL)
        
        for topic in topics:
            participants = set()
            contributors = defaultdict(int)
            
            for speaker, content in speaker_segments:
                content_lower = content.lower()
                
                # Check if speaker mentioned topic
                topic_mentioned = False
                for keyword in topic.keywords:
                    if keyword in content_lower:
                        topic_mentioned = True
                        contributors[speaker] += 1
                        
                for phrase in topic.key_phrases:
                    if phrase in content_lower:
                        topic_mentioned = True
                        contributors[speaker] += 1
                        
                if topic_mentioned:
                    participants.add(speaker)
                    
            topic.participants = list(participants)
            
            # Identify main contributors
            if contributors:
                sorted_contributors = sorted(contributors.items(), key=lambda x: x[1], reverse=True)
                topic.main_contributors = [speaker for speaker, count in sorted_contributors[:3]]
                
    def get_status(self) -> Dict[str, Any]:
        """Get identifier status"""
        return {
            "config": self.config.to_dict(),
            "sklearn_available": self.is_sklearn_available,
            "topic_indicators": len(self.topic_indicators),
            "decision_indicators": len(self.decision_indicators)
        }


# Factory functions
def create_meeting_topic_identifier() -> TopicIdentifier:
    """Create identifier for general meetings"""
    config = TopicConfig(
        max_topics=5,
        min_relevance_score=0.3,
        min_mentions=2,
        use_clustering=True,
        include_speaker_analysis=True
    )
    return TopicIdentifier(config)


def create_discussion_topic_identifier() -> TopicIdentifier:
    """Create identifier for detailed discussions"""
    config = TopicConfig(
        max_topics=8,
        min_relevance_score=0.2,
        min_mentions=1,
        use_clustering=True,
        n_clusters=6,
        include_speaker_analysis=True
    )
    return TopicIdentifier(config)


def create_simple_topic_identifier() -> TopicIdentifier:
    """Create simple identifier without clustering"""
    config = TopicConfig(
        max_topics=3,
        min_relevance_score=0.4,
        min_mentions=3,
        use_clustering=False,
        include_speaker_analysis=False
    )
    return TopicIdentifier(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Topic Identifier Test")
    parser.add_argument("--transcript", type=str, required=True, help="Meeting transcript")
    parser.add_argument("--max-topics", type=int, default=5, help="Maximum topics")
    parser.add_argument("--min-relevance", type=float, default=0.3, help="Minimum relevance")
    args = parser.parse_args()
    
    # Create identifier
    identifier = create_meeting_topic_identifier()
    identifier.config.max_topics = args.max_topics
    identifier.config.min_relevance_score = args.min_relevance
    
    try:
        print(f"Identifier status: {identifier.get_status()}")
        
        # Identify topics
        print(f"Identifying topics from transcript...")
        result = identifier.identify_topics(args.transcript)
        
        if result.success:
            print(f"Topic identification completed successfully!")
            print(f"Topics found: {result.total_topics_found}")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Keywords extracted: {result.keywords_extracted}")
            print(f"Phrases identified: {result.phrases_identified}")
            
            print(f"\nIdentified Topics:")
            for i, topic in enumerate(result.topics, 1):
                print(f"{i}. {topic.name}")
                print(f"   Description: {topic.description}")
                print(f"   Importance: {topic.importance.value}")
                print(f"   Relevance: {topic.relevance_score:.3f}")
                print(f"   Mentions: {topic.mention_count}")
                print(f"   Time spent: {topic.time_spent:.1f}%")
                if topic.keywords:
                    print(f"   Keywords: {', '.join(topic.keywords[:5])}")
                if topic.participants:
                    print(f"   Participants: {', '.join(topic.participants)}")
                print()
                
        else:
            print(f"Topic identification failed: {result.error_message}")
            
    except Exception as e:
        print(f"Error: {e}")