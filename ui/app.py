"""
Main Streamlit application for AI Knowledge Search Platform.
Provides interactive UI for querying and visualizing the RAG pipeline.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Search Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .routing-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
    .optimization-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """Make API request to backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

def display_header():
    """Display application header."""
    st.markdown('<h1 class="main-header">🔍 AI Knowledge Search Platform</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Intelligent document search with query routing and context optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with configuration and stats."""
    st.sidebar.title("Configuration")
    
    # API Status
    st.sidebar.subheader("API Status")
    health_response = make_api_request("/health")
    
    if "error" not in health_response:
        st.sidebar.success("✅ API Connected")
        
        # Display basic stats
        stats_response = make_api_request("/stats")
        if "error" not in stats_response:
            pipeline_stats = stats_response.get("pipeline_stats", {})
            collection_stats = pipeline_stats.get("collection_stats", {})
            
            st.sidebar.metric("Documents", collection_stats.get("total_documents", 0))
            st.sidebar.metric("Collection", collection_stats.get("collection_name", "N/A"))
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.error(f"Error: {health_response['error']}")
    
    # Query Configuration
    st.sidebar.subheader("Query Settings")
    max_results = st.sidebar.slider("Max Results", 1, 20, 10)
    enable_optimization = st.sidebar.checkbox("Enable Optimization", value=True)
    
    return max_results, enable_optimization

def display_query_interface():
    """Display main query interface."""
    st.subheader("🔍 Search Query")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is machine learning? How does neural network work?",
        help="Ask any question about your documents. The system will intelligently route your query and provide an optimized response."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    return query, search_button

def display_routing_visualizer(routing_decision: Dict[str, Any]):
    """Display query routing decision visualization."""
    st.subheader("🎯 Query Routing Decision")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="routing-card">', unsafe_allow_html=True)
        
        # Intent information
        intent = routing_decision.get("intent", {})
        st.write(f"**Detected Intent:** {intent.get('type', 'unknown').title()}")
        st.write(f"**Confidence:** {intent.get('confidence', 0):.2f}")
        st.write(f"**Description:** {intent.get('description', 'N/A')}")
        
        # Strategy information
        strategy = routing_decision.get("strategy", {})
        st.write(f"**Chosen Strategy:** {strategy.get('type', 'unknown').title()}")
        st.write(f"**Description:** {strategy.get('description', 'N/A')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Visual representation
        intent_type = intent.get('type', 'unknown')
        strategy_type = strategy.get('type', 'unknown')
        
        # Create a simple flow diagram
        fig = go.Figure()
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=[0, 1, 2],
            y=[0, 0, 0],
            mode='markers+text',
            marker=dict(size=50, color=['#ff6b6b', '#4ecdc4', '#45b7d1']),
            text=[f"Query<br>{intent_type}", f"Route<br>{strategy_type}", "Retrieve"],
            textposition="middle center",
            textfont=dict(size=12, color="white")
        ))
        
        # Add arrows
        fig.add_annotation(
            x=0.5, y=0,
            ax=0.2, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#666"
        )
        
        fig.add_annotation(
            x=1.5, y=0,
            ax=1.2, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#666"
        )
        
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            width=300, height=150,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_context_inspector(optimization: Dict[str, Any]):
    """Display context optimization results."""
    st.subheader("⚡ Context Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="optimization-card">', unsafe_allow_html=True)
        
        original_count = optimization.get("original_count", 0)
        optimized_count = optimization.get("optimized_count", 0)
        
        st.write(f"**Original Results:** {original_count}")
        st.write(f"**Optimized Results:** {optimized_count}")
        
        if original_count > 0:
            reduction_percent = ((original_count - optimized_count) / original_count) * 100
            st.write(f"**Reduction:** {reduction_percent:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Optimization stats
        stats = optimization.get("stats", {})
        if stats:
            dedup_stats = stats.get("deduplication", {})
            comp_stats = stats.get("compression", {})
            
            if dedup_stats:
                st.write(f"**Token Savings:** {dedup_stats.get('token_savings', 0)}")
                st.write(f"**Token Reduction:** {dedup_stats.get('token_reduction_percent', 0):.1f}%")
            
            if comp_stats:
                st.write(f"**Compression Ratio:** {comp_stats.get('compression_ratio', 1):.2f}")

def display_search_results(rag_response: Dict[str, Any]):
    """Display search results and answer."""
    st.subheader("📝 Answer")
    
    # Display LLM response
    llm_response = rag_response.get("llm_response", {})
    answer = llm_response.get("text", "No answer generated")
    
    st.write(answer)
    
    # Display metadata
    with st.expander("📊 Response Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**LLM Provider:**", llm_response.get("provider", "N/A"))
            st.write("**Model:**", llm_response.get("model", "N/A"))
            st.write("**Context Sources:**", llm_response.get("context_sources", 0))
            st.write("**Context Tokens:**", llm_response.get("context_tokens", 0))
        
        with col2:
            usage = llm_response.get("usage", {})
            if usage:
                st.write("**Prompt Tokens:**", usage.get("prompt_tokens", 0))
                st.write("**Completion Tokens:**", usage.get("completion_tokens", 0))
                st.write("**Total Tokens:**", usage.get("total_tokens", 0))
        
        st.write("**Processing Time:**", f"{rag_response.get('processing_time', 0):.2f}s")

def display_source_documents(retrieval_results: Dict[str, Any]):
    """Display source documents used for the answer."""
    st.subheader("📚 Source Documents")
    
    # This would display the actual retrieved documents
    # For now, show a placeholder
    st.info("Source documents would be displayed here in a full implementation.")

def main():
    """Main Streamlit application."""
    display_header()
    
    # Sidebar
    max_results, enable_optimization = display_sidebar()
    
    # Main content
    query, search_button = display_query_interface()
    
    if search_button and query:
        with st.spinner("Processing your query..."):
            # Make API request
            request_data = {
                "query": query,
                "max_results": max_results
            }
            
            response = make_api_request("/query/rag", "POST", request_data)
            
            if "error" not in response:
                # Display results
                st.success("✅ Query processed successfully!")
                
                # Display routing decision
                routing_decision = response.get("routing_decision", {})
                if routing_decision:
                    display_routing_visualizer(routing_decision)
                
                # Display optimization results
                optimization = response.get("optimization", {})
                if optimization:
                    display_context_inspector(optimization)
                
                # Display search results
                display_search_results(response)
                
                # Display source documents
                retrieval_results = response.get("retrieval_results", {})
                if retrieval_results:
                    display_source_documents(retrieval_results)
                
            else:
                st.error(f"❌ Error processing query: {response['error']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        AI Knowledge Search Platform | Built with FastAPI, Streamlit, and ChromaDB
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
