"""
Intent classifier for determining query type and routing strategy.
Uses hybrid approach: keyword patterns + optional ML classification.
"""

import re
from typing import Dict, Any, List, Tuple
from enum import Enum
from ..core.logger import app_logger
from ..core.utils import clean_text

class QueryIntent(Enum):
    """Query intent types."""
    FACTOID = "factoid"          # What is X? Who is Y?
    SEMANTIC = "semantic"        # Find similar concepts
    ANALYTICAL = "analytical"    # Compare, analyze, explain
    CONVERSATIONAL = "conversational"  # How to, help me, guide

class IntentClassifier:
    """Classifies query intent using pattern matching and optional ML."""
    
    def __init__(self):
        self.logger = app_logger
        self.patterns = self._initialize_patterns()
        self.confidence_threshold = 0.6
    
    def _initialize_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Initialize keyword patterns for each intent type."""
        return {
            QueryIntent.FACTOID: [
                # Direct question patterns
                r'\b(what|who|when|where|which|how many|how much)\b',
                r'\b(define|definition|meaning|is|are|was|were)\b',
                r'\b(explain|tell me about|describe)\b',
                # Specific fact patterns
                r'\b(name|list|show me|give me)\b',
                r'\b(facts?|information|details?)\b'
            ],
            
            QueryIntent.SEMANTIC: [
                # Similarity and related concepts
                r'\b(similar|like|related|associated|connected)\b',
                r'\b(find|search|look for|discover)\b',
                r'\b(concepts?|ideas?|topics?|themes?)\b',
                r'\b(patterns?|trends?|relationships?)\b',
                r'\b(compare|contrast|versus|vs)\b'
            ],
            
            QueryIntent.ANALYTICAL: [
                # Analysis and comparison
                r'\b(analyze|analysis|examine|investigate)\b',
                r'\b(compare|comparison|contrast|difference)\b',
                r'\b(evaluate|assess|judge|critique)\b',
                r'\b(pros? and cons?|advantages?|disadvantages?)\b',
                r'\b(why|how|because|reason|cause|effect)\b',
                r'\b(implications?|consequences?|impact)\b'
            ],
            
            QueryIntent.CONVERSATIONAL: [
                # Help and guidance
                r'\b(how to|how do i|how can i)\b',
                r'\b(help|assist|guide|tutorial|steps?)\b',
                r'\b(show me how|walk me through)\b',
                r'\b(instructions?|directions?|procedure)\b',
                r'\b(best practices?|recommendations?)\b',
                r'\b(troubleshoot|fix|solve|resolve)\b'
            ]
        }
    
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float, Dict[str, Any]]:
        """
        Classify query intent and return confidence score.
        
        Args:
            query: Input query text
            
        Returns:
            Tuple of (intent, confidence_score, explanation)
        """
        if not query or not query.strip():
            return QueryIntent.FACTOID, 0.0, {"reason": "empty_query"}
        
        query = clean_text(query).lower()
        scores = {}
        explanations = {}
        
        # Calculate pattern-based scores
        for intent, patterns in self.patterns.items():
            score = self._calculate_pattern_score(query, patterns)
            scores[intent] = score
            
            # Find matching patterns for explanation
            matching_patterns = []
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    matching_patterns.append(pattern)
            
            explanations[intent] = {
                "pattern_score": score,
                "matching_patterns": matching_patterns,
                "pattern_count": len(matching_patterns)
            }
        
        # Find best intent
        best_intent = max(scores.items(), key=lambda x: x[1])
        intent, confidence = best_intent
        
        # Additional heuristics
        confidence = self._apply_heuristics(query, intent, confidence)
        
        # Create explanation
        explanation = {
            "intent": intent.value,
            "confidence": confidence,
            "scores": {k.value: v for k, v in scores.items()},
            "explanations": {k.value: v for k, v in explanations.items()},
            "query_length": len(query),
            "query_words": len(query.split())
        }
        
        self.logger.debug(f"Intent classification: '{query}' -> {intent.value} ({confidence:.2f})")
        
        return intent, confidence, explanation
    
    def _calculate_pattern_score(self, query: str, patterns: List[str]) -> float:
        """Calculate pattern matching score for a query."""
        total_matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                total_matches += 1
        
        # Base score from pattern matches
        base_score = total_matches / total_patterns if total_patterns > 0 else 0
        
        # Bonus for multiple matches
        if total_matches > 1:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _apply_heuristics(self, query: str, intent: QueryIntent, confidence: float) -> float:
        """Apply additional heuristics to improve confidence."""
        
        # Question mark bonus
        if '?' in query:
            confidence += 0.1
        
        # Length-based adjustments
        word_count = len(query.split())
        
        if word_count < 3:
            # Very short queries are likely factoid
            if intent == QueryIntent.FACTOID:
                confidence += 0.1
            else:
                confidence -= 0.1
        elif word_count > 15:
            # Long queries are likely analytical or conversational
            if intent in [QueryIntent.ANALYTICAL, QueryIntent.CONVERSATIONAL]:
                confidence += 0.1
        
        # Specific keyword bonuses
        if intent == QueryIntent.FACTOID:
            if any(word in query for word in ['what', 'who', 'when', 'where']):
                confidence += 0.15
        
        elif intent == QueryIntent.CONVERSATIONAL:
            if any(phrase in query for phrase in ['how to', 'how do', 'help me']):
                confidence += 0.15
        
        elif intent == QueryIntent.ANALYTICAL:
            if any(word in query for word in ['analyze', 'compare', 'evaluate']):
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def get_intent_description(self, intent: QueryIntent) -> str:
        """Get human-readable description of intent."""
        descriptions = {
            QueryIntent.FACTOID: "Direct factual questions seeking specific information",
            QueryIntent.SEMANTIC: "Conceptual queries looking for related or similar content",
            QueryIntent.ANALYTICAL: "Complex queries requiring analysis, comparison, or explanation",
            QueryIntent.CONVERSATIONAL: "Help-seeking queries requesting guidance or instructions"
        }
        return descriptions.get(intent, "Unknown intent")
    
    def classify_batch(self, queries: List[str]) -> List[Tuple[QueryIntent, float, Dict[str, Any]]]:
        """Classify multiple queries in batch."""
        results = []
        for query in queries:
            result = self.classify_intent(query)
            results.append(result)
        return results
