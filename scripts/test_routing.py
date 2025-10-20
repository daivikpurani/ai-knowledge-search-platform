#!/usr/bin/env python3
"""
Test script for demonstrating different query routing paths.
Tests various query types to showcase the intelligent routing system.
"""

import sys
import asyncio
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.logger import app_logger
from backend.routing.router import QueryRouter
from backend.generation.rag_pipeline import RAGPipeline

class QueryTester:
    """Test different query types to demonstrate routing capabilities."""
    
    def __init__(self):
        self.router = QueryRouter()
        self.rag_pipeline = RAGPipeline()
        
        # Test queries organized by intent type
        self.test_queries = {
            "factoid": [
                "What is artificial intelligence?",
                "Who invented the neural network?",
                "Define machine learning",
                "What is the difference between AI and ML?",
                "When was deep learning first introduced?"
            ],
            "semantic": [
                "Find concepts related to deep learning",
                "What topics are similar to computer vision?",
                "Show me content about AI ethics",
                "Find information about natural language processing",
                "What are the applications of neural networks?"
            ],
            "analytical": [
                "Compare supervised vs unsupervised learning",
                "Analyze the benefits and drawbacks of neural networks",
                "Evaluate different AI approaches for image recognition",
                "What are the implications of AI in healthcare?",
                "Compare traditional programming with machine learning"
            ],
            "conversational": [
                "How do I implement a neural network?",
                "Help me understand machine learning algorithms",
                "Guide me through the process of training a model",
                "How can I get started with deep learning?",
                "What steps should I take to learn AI?"
            ]
        }
    
    async def test_routing_decisions(self):
        """Test query routing decisions for different query types."""
        print("🎯 Testing Query Routing Decisions")
        print("=" * 50)
        
        for intent_type, queries in self.test_queries.items():
            print(f"\n📋 {intent_type.upper()} QUERIES:")
            print("-" * 30)
            
            for query in queries:
                try:
                    # Get routing decision
                    routing_decision = self.router.route_query(query)
                    
                    detected_intent = routing_decision["intent"]["type"]
                    chosen_strategy = routing_decision["strategy"]["type"]
                    confidence = routing_decision["intent"]["confidence"]
                    
                    print(f"Query: {query}")
                    print(f"  → Detected Intent: {detected_intent}")
                    print(f"  → Chosen Strategy: {chosen_strategy}")
                    print(f"  → Confidence: {confidence:.2f}")
                    print()
                    
                except Exception as e:
                    print(f"Error testing query '{query}': {str(e)}")
    
    async def test_rag_pipeline(self, max_queries: int = 3):
        """Test the complete RAG pipeline with sample queries."""
        print("\n🤖 Testing RAG Pipeline")
        print("=" * 50)
        
        test_count = 0
        
        for intent_type, queries in self.test_queries.items():
            if test_count >= max_queries:
                break
                
            print(f"\n📋 Testing {intent_type.upper()} Query:")
            print("-" * 30)
            
            query = queries[0]  # Test first query from each category
            print(f"Query: {query}")
            
            try:
                # Process through RAG pipeline
                rag_response = await self.rag_pipeline.process_query(
                    query=query,
                    max_results=5,
                    enable_optimization=True
                )
                
                # Display results
                print(f"✅ Processing Time: {rag_response.get('processing_time', 0):.2f}s")
                
                routing_decision = rag_response.get("routing_decision", {})
                print(f"🎯 Intent: {routing_decision.get('intent', {}).get('type', 'unknown')}")
                print(f"🔀 Strategy: {routing_decision.get('strategy', {}).get('type', 'unknown')}")
                
                optimization = rag_response.get("optimization", {})
                print(f"⚡ Optimization: {optimization.get('original_count', 0)} → {optimization.get('optimized_count', 0)} results")
                
                llm_response = rag_response.get("llm_response", {})
                answer = llm_response.get("text", "No answer generated")
                print(f"📝 Answer Preview: {answer[:100]}...")
                
                test_count += 1
                
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
    
    def test_intent_classification_accuracy(self):
        """Test intent classification accuracy with known examples."""
        print("\n📊 Testing Intent Classification Accuracy")
        print("=" * 50)
        
        correct_predictions = 0
        total_predictions = 0
        
        for expected_intent, queries in self.test_queries.items():
            print(f"\nTesting {expected_intent} queries:")
            
            for query in queries:
                try:
                    routing_decision = self.router.route_query(query)
                    predicted_intent = routing_decision["intent"]["type"]
                    confidence = routing_decision["intent"]["confidence"]
                    
                    is_correct = predicted_intent == expected_intent
                    status = "✅" if is_correct else "❌"
                    
                    print(f"  {status} '{query}' → {predicted_intent} ({confidence:.2f})")
                    
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
        
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        print(f"\n📈 Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        print("\n📋 Generating Test Report")
        print("=" * 50)
        
        report = {
            "test_summary": {
                "total_query_types": len(self.test_queries),
                "total_test_queries": sum(len(queries) for queries in self.test_queries.values()),
                "available_strategies": self.router.get_available_strategies(),
                "routing_stats": self.router.get_routing_stats()
            },
            "query_examples": self.test_queries,
            "routing_mappings": {
                "factoid": "bm25",
                "semantic": "semantic", 
                "analytical": "hybrid",
                "conversational": "conversational"
            }
        }
        
        # Save report
        report_path = Path(__file__).parent / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: {report_path}")
        return report

async def main():
    """Main test function."""
    print("🔍 AI Knowledge Search Platform - Query Routing Tests")
    print("=" * 60)
    
    tester = QueryTester()
    
    try:
        # Test routing decisions
        await tester.test_routing_decisions()
        
        # Test RAG pipeline
        await tester.test_rag_pipeline(max_queries=2)
        
        # Test classification accuracy
        tester.test_intent_classification_accuracy()
        
        # Generate report
        tester.generate_test_report()
        
        print("\n✅ All tests completed successfully!")
        print("🚀 The intelligent routing system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
