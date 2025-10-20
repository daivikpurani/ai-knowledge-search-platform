"""
Routing visualizer component for Streamlit UI.
Displays query routing decisions and strategy selection.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

class RoutingVisualizer:
    """Component for visualizing query routing decisions."""
    
    def __init__(self):
        self.intent_colors = {
            "factoid": "#ff6b6b",
            "semantic": "#4ecdc4", 
            "analytical": "#45b7d1",
            "conversational": "#96ceb4"
        }
        
        self.strategy_colors = {
            "bm25": "#ff9ff3",
            "semantic": "#54a0ff",
            "hybrid": "#5f27cd",
            "conversational": "#00d2d3"
        }
    
    def display_routing_decision(self, routing_decision: Dict[str, Any]):
        """Display complete routing decision visualization."""
        st.subheader("🎯 Query Routing Decision")
        
        # Main routing card
        self._display_routing_card(routing_decision)
        
        # Visual flow diagram
        self._display_routing_flow(routing_decision)
        
        # Detailed explanation
        self._display_routing_explanation(routing_decision)
    
    def _display_routing_card(self, routing_decision: Dict[str, Any]):
        """Display routing decision in a card format."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background-color: #e8f4fd; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #1f77b4;">
            """, unsafe_allow_html=True)
            
            # Intent information
            intent = routing_decision.get("intent", {})
            intent_type = intent.get("type", "unknown")
            intent_confidence = intent.get("confidence", 0)
            
            st.write(f"**🎯 Detected Intent:** {intent_type.title()}")
            st.write(f"**📊 Confidence:** {intent_confidence:.2f}")
            st.write(f"**📝 Description:** {intent.get('description', 'N/A')}")
            
            # Strategy information
            strategy = routing_decision.get("strategy", {})
            strategy_type = strategy.get("type", "unknown")
            
            st.write(f"**🔀 Chosen Strategy:** {strategy_type.title()}")
            st.write(f"**📋 Description:** {strategy.get('description', 'N/A')}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Confidence gauge
            self._display_confidence_gauge(intent_confidence)
    
    def _display_confidence_gauge(self, confidence: float):
        """Display confidence as a gauge chart."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence"},
            delta = {'reference': 0.8},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.8], 'color': "yellow"},
                    {'range': [0.8, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_routing_flow(self, routing_decision: Dict[str, Any]):
        """Display routing flow diagram."""
        st.subheader("🔄 Routing Flow")
        
        intent = routing_decision.get("intent", {})
        strategy = routing_decision.get("strategy", {})
        
        intent_type = intent.get("type", "unknown")
        strategy_type = strategy.get("type", "unknown")
        
        # Create flow diagram
        fig = go.Figure()
        
        # Define positions
        positions = {
            "query": (0, 0),
            "intent": (1, 0),
            "strategy": (2, 0),
            "retrieve": (3, 0)
        }
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=[pos[0] for pos in positions.values()],
            y=[pos[1] for pos in positions.values()],
            mode='markers+text',
            marker=dict(
                size=60,
                color=[
                    self.intent_colors.get(intent_type, "#666"),
                    self.intent_colors.get(intent_type, "#666"),
                    self.strategy_colors.get(strategy_type, "#666"),
                    "#95a5a6"
                ]
            ),
            text=[
                f"Query<br>{intent_type}",
                f"Intent<br>{intent_type}",
                f"Strategy<br>{strategy_type}",
                "Retrieve"
            ],
            textposition="middle center",
            textfont=dict(size=10, color="white")
        ))
        
        # Add arrows
        for i in range(len(positions) - 1):
            fig.add_annotation(
                x=positions[list(positions.keys())[i+1]][0] - 0.2,
                y=0,
                ax=positions[list(positions.keys())[i]][0] + 0.2,
                ay=0,
                xref="x", yref="y",
                axref="x", ayref="y",
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#666"
            )
        
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            height=150,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_routing_explanation(self, routing_decision: Dict[str, Any]):
        """Display detailed routing explanation."""
        with st.expander("🔍 Routing Explanation"):
            explanation = routing_decision.get("explanation", {})
            
            if explanation:
                # Display scores for each intent
                scores = explanation.get("scores", {})
                if scores:
                    st.write("**Intent Classification Scores:**")
                    
                    # Create bar chart of intent scores
                    intent_names = list(scores.keys())
                    intent_values = list(scores.values())
                    
                    fig = px.bar(
                        x=intent_names,
                        y=intent_values,
                        title="Intent Classification Scores",
                        color=intent_values,
                        color_continuous_scale="Blues"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Intent Type",
                        yaxis_title="Score",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display matching patterns
                explanations = explanation.get("explanations", {})
                if explanations:
                    st.write("**Pattern Matching Details:**")
                    
                    for intent_type, details in explanations.items():
                        with st.expander(f"{intent_type.title()} Intent"):
                            st.write(f"**Pattern Score:** {details.get('pattern_score', 0):.2f}")
                            st.write(f"**Matching Patterns:** {details.get('pattern_count', 0)}")
                            
                            matching_patterns = details.get("matching_patterns", [])
                            if matching_patterns:
                                st.write("**Matched Patterns:**")
                                for pattern in matching_patterns:
                                    st.code(pattern)
    
    def display_strategy_comparison(self, strategies: List[Dict[str, Any]]):
        """Display comparison of available strategies."""
        st.subheader("⚖️ Strategy Comparison")
        
        if not strategies:
            st.info("No strategy information available")
            return
        
        # Create comparison table
        comparison_data = []
        for strategy in strategies:
            comparison_data.append({
                "Strategy": strategy.get("name", "Unknown"),
                "Description": strategy.get("description", "N/A"),
                "Max Results": strategy.get("max_results", 0),
                "Use Cases": ", ".join(strategy.get("use_cases", []))
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    def display_routing_stats(self, routing_stats: Dict[str, Any]):
        """Display routing statistics."""
        st.subheader("📊 Routing Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Available Strategies",
                routing_stats.get("available_strategies", 0)
            )
        
        with col2:
            st.metric(
                "Intent Types",
                routing_stats.get("intent_types", 0)
            )
        
        with col3:
            st.metric(
                "Confidence Threshold",
                f"{routing_stats.get('confidence_threshold', 0):.2f}"
            )
        
        # Display strategy mappings
        strategy_mappings = routing_stats.get("strategy_mappings", {})
        if strategy_mappings:
            st.write("**Strategy Mappings:**")
            for intent, strategy in strategy_mappings.items():
                st.write(f"- {intent.title()} → {strategy.title()}")
    
    def display_intent_examples(self):
        """Display examples of different intent types."""
        st.subheader("💡 Intent Examples")
        
        examples = {
            "Factoid": [
                "What is machine learning?",
                "Who invented the neural network?",
                "Define artificial intelligence"
            ],
            "Semantic": [
                "Find concepts related to deep learning",
                "What topics are similar to computer vision?",
                "Show me content about AI ethics"
            ],
            "Analytical": [
                "Compare supervised vs unsupervised learning",
                "Analyze the benefits of neural networks",
                "Evaluate different AI approaches"
            ],
            "Conversational": [
                "How do I implement a neural network?",
                "Help me understand machine learning",
                "Guide me through AI concepts"
            ]
        }
        
        for intent_type, query_examples in examples.items():
            with st.expander(f"{intent_type} Intent Examples"):
                for example in query_examples:
                    st.write(f"• {example}")
