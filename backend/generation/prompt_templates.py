"""
Prompt templates for different query types and intents.
Provides context-aware prompts for better LLM responses.
"""

from typing import Dict, Any, List
from enum import Enum

class QueryIntent(Enum):
    """Query intent types for prompt selection."""
    FACTOID = "factoid"
    SEMANTIC = "semantic"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"

class PromptTemplates:
    """Template manager for different query types and contexts."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize prompt templates for different intents."""
        return {
            QueryIntent.FACTOID.value: {
                "system": "You are a helpful AI assistant that provides accurate, factual information based on the given context. Focus on providing direct, clear answers to factual questions.",
                "user": """Context:
{context}

Question: {query}

Please provide a direct, factual answer based on the context above. If the context doesn't contain the specific information needed, please state that clearly.""",
                "follow_up": "Based on the context, here are the key facts:"
            },
            
            QueryIntent.SEMANTIC.value: {
                "system": "You are a helpful AI assistant that helps users find and understand related concepts and information. Focus on explaining relationships and connections between ideas.",
                "user": """Context:
{context}

Query: {query}

Please help me understand the concepts related to this query based on the context above. Explain the relationships and connections you find.""",
                "follow_up": "Here are the related concepts and their relationships:"
            },
            
            QueryIntent.ANALYTICAL.value: {
                "system": "You are a helpful AI assistant that provides analytical insights and comparisons. Focus on breaking down complex topics, comparing different aspects, and providing thoughtful analysis.",
                "user": """Context:
{context}

Analytical Query: {query}

Please provide a thorough analysis based on the context above. Break down the topic, compare different aspects, and provide insights and conclusions.""",
                "follow_up": "Here is my analysis based on the available context:"
            },
            
            QueryIntent.CONVERSATIONAL.value: {
                "system": "You are a helpful AI assistant that provides guidance and step-by-step instructions. Focus on being conversational, helpful, and providing actionable advice.",
                "user": """Context:
{context}

Question: {query}

Please provide helpful guidance and step-by-step instructions based on the context above. Be conversational and practical in your response.""",
                "follow_up": "Here's how you can approach this:"
            }
        }
    
    def get_prompt(self, intent: str, query: str, context: str, 
                  include_system: bool = True) -> str:
        """
        Get formatted prompt for a specific intent.
        
        Args:
            intent: Query intent type
            query: User query
            context: Retrieved context
            include_system: Whether to include system message
            
        Returns:
            Formatted prompt string
        """
        try:
            template = self.templates.get(intent, self.templates[QueryIntent.FACTOID.value])
            
            if include_system:
                system_msg = template["system"]
                user_msg = template["user"].format(context=context, query=query)
                return f"{system_msg}\n\n{user_msg}"
            else:
                return template["user"].format(context=context, query=query)
                
        except Exception as e:
            # Fallback to factoid template
            template = self.templates[QueryIntent.FACTOID.value]
            return template["user"].format(context=context, query=query)
    
    def get_system_message(self, intent: str) -> str:
        """Get system message for a specific intent."""
        template = self.templates.get(intent, self.templates[QueryIntent.FACTOID.value])
        return template["system"]
    
    def get_follow_up_template(self, intent: str) -> str:
        """Get follow-up template for a specific intent."""
        template = self.templates.get(intent, self.templates[QueryIntent.FACTOID.value])
        return template["follow_up"]
    
    def create_custom_prompt(self, query: str, context: str, 
                           instructions: str = "") -> str:
        """
        Create a custom prompt with specific instructions.
        
        Args:
            query: User query
            context: Retrieved context
            instructions: Custom instructions
            
        Returns:
            Custom formatted prompt
        """
        base_instructions = "You are a helpful AI assistant that provides accurate and helpful responses based on the given context."
        
        if instructions:
            system_msg = f"{base_instructions} {instructions}"
        else:
            system_msg = base_instructions
        
        user_msg = f"""Context:
{context}

Question: {query}

Please provide a helpful response based on the context above."""
        
        return f"{system_msg}\n\n{user_msg}"
    
    def get_context_summary_prompt(self, context: str, max_length: int = 500) -> str:
        """Get prompt for summarizing context."""
        return f"""Please summarize the following context in {max_length} words or less, focusing on the key points and main ideas:

{context}

Summary:"""
    
    def get_query_expansion_prompt(self, query: str) -> str:
        """Get prompt for expanding a query with related terms."""
        return f"""Given this query: "{query}"

Please suggest 3-5 related terms or phrases that could help find more relevant information. Focus on synonyms, related concepts, and alternative phrasings.

Related terms:"""
    
    def get_answer_validation_prompt(self, query: str, answer: str, context: str) -> str:
        """Get prompt for validating answer quality."""
        return f"""Query: {query}

Answer: {answer}

Context: {context}

Please evaluate this answer on a scale of 1-10 based on:
1. Accuracy (is it correct based on the context?)
2. Completeness (does it fully address the query?)
3. Clarity (is it easy to understand?)

Provide a score and brief explanation."""
    
    def get_intent_explanation_prompt(self, query: str, intent: str) -> str:
        """Get prompt for explaining intent classification."""
        return f"""Query: "{query}"
Classified Intent: {intent}

Please explain why this query was classified as "{intent}" and what type of response would be most appropriate."""
    
    def list_available_intents(self) -> List[str]:
        """Get list of available intent types."""
        return list(self.templates.keys())
    
    def get_template_info(self, intent: str) -> Dict[str, str]:
        """Get information about a specific template."""
        return self.templates.get(intent, {})
    
    def add_custom_template(self, intent: str, system: str, user: str, follow_up: str = ""):
        """Add a custom template for a new intent."""
        self.templates[intent] = {
            "system": system,
            "user": user,
            "follow_up": follow_up
        }
    
    def update_template(self, intent: str, template_type: str, new_content: str):
        """Update a specific part of a template."""
        if intent in self.templates and template_type in self.templates[intent]:
            self.templates[intent][template_type] = new_content
        else:
            raise ValueError(f"Template {intent} or type {template_type} not found")
