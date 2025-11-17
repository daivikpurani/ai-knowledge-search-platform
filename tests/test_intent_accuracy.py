"""
Test suite for verifying intent classification accuracy.
Tests that the system achieves at least 85% accuracy.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from backend.routing.router import QueryRouter
from backend.routing.intent_classifier import QueryIntent

class TestIntentAccuracy:
    """Test intent classification accuracy."""
    
    @pytest.fixture
    def router(self):
        """Create router instance."""
        return QueryRouter()
    
    @pytest.fixture
    def test_queries(self):
        """Test queries organized by expected intent."""
        return {
            QueryIntent.FACTOID: [
                "What is artificial intelligence?",
                "Who invented the neural network?",
                "Define machine learning",
                "What is the difference between AI and ML?",
                "When was deep learning first introduced?",
                "What does NLP stand for?",
                "Who created the transformer architecture?",
                "What is a convolutional neural network?",
                "Define reinforcement learning",
                "What is the purpose of backpropagation?",
                "Name three types of machine learning",
                "What is supervised learning?",
                "Explain what a neural network is",
                "What are the components of a neural network?",
                "Who is considered the father of AI?"
            ],
            
            QueryIntent.SEMANTIC: [
                "Find concepts related to deep learning",
                "What topics are similar to computer vision?",
                "Show me content about AI ethics",
                "Find information about natural language processing",
                "What are the applications of neural networks?",
                "Discover topics related to machine learning",
                "Find content similar to reinforcement learning",
                "What concepts are associated with transformers?",
                "Show me related topics to computer vision",
                "Find information about similar AI techniques",
                "What topics connect to deep learning?",
                "Discover concepts related to NLP",
                "Find content about similar neural architectures",
                "What are related fields to machine learning?",
                "Show me topics associated with AI research"
            ],
            
            QueryIntent.ANALYTICAL: [
                "Compare supervised vs unsupervised learning",
                "Analyze the benefits and drawbacks of neural networks",
                "Evaluate different AI approaches for image recognition",
                "What are the implications of AI in healthcare?",
                "Compare traditional programming with machine learning",
                "Analyze the effectiveness of different optimization algorithms",
                "Evaluate the pros and cons of deep learning",
                "Compare CNN vs RNN architectures",
                "What are the trade-offs of using transformers?",
                "Analyze the impact of AI on society",
                "Compare different neural network architectures",
                "Evaluate the performance of various ML algorithms",
                "What are the advantages and disadvantages of reinforcement learning?",
                "Analyze the relationship between AI and automation",
                "Compare different approaches to natural language understanding"
            ],
            
            QueryIntent.CONVERSATIONAL: [
                "How do I implement a neural network?",
                "Help me understand machine learning algorithms",
                "Guide me through the process of training a model",
                "How can I get started with deep learning?",
                "What steps should I take to learn AI?",
                "How do I train a machine learning model?",
                "Can you help me understand backpropagation?",
                "What's the best way to learn neural networks?",
                "How do I implement gradient descent?",
                "Guide me through building an AI project",
                "How can I improve my model's accuracy?",
                "What are the steps to preprocess data?",
                "How do I choose the right algorithm?",
                "Can you help me debug my neural network?",
                "What's the process for deploying a model?"
            ]
        }
    
    def test_intent_classification_accuracy(self, router, test_queries):
        """Test that intent classification achieves at least 85% accuracy."""
        correct_predictions = 0
        total_predictions = 0
        
        for expected_intent, queries in test_queries.items():
            for query in queries:
                routing_decision = router.route_query(query)
                predicted_intent = QueryIntent(routing_decision["intent"]["type"])
                
                if predicted_intent == expected_intent:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        print(f"\nIntent Classification Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        
        # Assert minimum 85% accuracy
        assert accuracy >= 85.0, f"Intent classification accuracy ({accuracy:.2f}%) is below required 85%"
    
    def test_factoid_queries(self, router, test_queries):
        """Test factoid query classification."""
        factoid_queries = test_queries[QueryIntent.FACTOID]
        correct = 0
        
        for query in factoid_queries:
            routing_decision = router.route_query(query)
            if routing_decision["intent"]["type"] == "factoid":
                correct += 1
        
        accuracy = (correct / len(factoid_queries) * 100) if factoid_queries else 0
        assert accuracy >= 80.0, f"Factoid classification accuracy ({accuracy:.2f}%) is below 80%"
    
    def test_semantic_queries(self, router, test_queries):
        """Test semantic query classification."""
        semantic_queries = test_queries[QueryIntent.SEMANTIC]
        correct = 0
        
        for query in semantic_queries:
            routing_decision = router.route_query(query)
            if routing_decision["intent"]["type"] == "semantic":
                correct += 1
        
        accuracy = (correct / len(semantic_queries) * 100) if semantic_queries else 0
        assert accuracy >= 80.0, f"Semantic classification accuracy ({accuracy:.2f}%) is below 80%"
    
    def test_analytical_queries(self, router, test_queries):
        """Test analytical query classification."""
        analytical_queries = test_queries[QueryIntent.ANALYTICAL]
        correct = 0
        
        for query in analytical_queries:
            routing_decision = router.route_query(query)
            if routing_decision["intent"]["type"] == "analytical":
                correct += 1
        
        accuracy = (correct / len(analytical_queries) * 100) if analytical_queries else 0
        assert accuracy >= 80.0, f"Analytical classification accuracy ({accuracy:.2f}%) is below 80%"
    
    def test_conversational_queries(self, router, test_queries):
        """Test conversational query classification."""
        conversational_queries = test_queries[QueryIntent.CONVERSATIONAL]
        correct = 0
        
        for query in conversational_queries:
            routing_decision = router.route_query(query)
            if routing_decision["intent"]["type"] == "conversational":
                correct += 1
        
        accuracy = (correct / len(conversational_queries) * 100) if conversational_queries else 0
        assert accuracy >= 80.0, f"Conversational classification accuracy ({accuracy:.2f}%) is below 80%"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

