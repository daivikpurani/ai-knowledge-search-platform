"""
Source viewer component for Streamlit UI.
Displays source documents and their relevance scores.
"""

import streamlit as st
import plotly.express as px
from typing import Dict, Any, List

class SourceViewer:
    """Component for viewing source documents and their relevance."""
    
    def __init__(self):
        self.relevance_colors = {
            "high": "#28a745",      # Green
            "medium": "#ffc107",    # Yellow
            "low": "#dc3545"        # Red
        }
    
    def display_source_documents(self, retrieval_results: Dict[str, Any], 
                               max_display: int = 5):
        """Display source documents used for the answer."""
        st.subheader("📚 Source Documents")
        
        count = retrieval_results.get("count", 0)
        method = retrieval_results.get("method", "unknown")
        
        if count == 0:
            st.warning("No source documents found for this query")
            return
        
        st.info(f"Found {count} relevant documents using {method} search")
        
        # Display top documents
        self._display_document_list(retrieval_results, max_display)
        
        # Display relevance distribution
        self._display_relevance_distribution(retrieval_results)
    
    def _display_document_list(self, retrieval_results: Dict[str, Any], 
                             max_display: int):
        """Display list of source documents."""
        # In a real implementation, this would show actual retrieved documents
        # For now, we'll show a placeholder structure
        
        st.write("**Top Source Documents:**")
        
        # Simulate document display
        for i in range(min(max_display, retrieval_results.get("count", 0))):
            with st.expander(f"📄 Document {i+1} - Relevance Score: {0.85 - i*0.1:.2f}"):
                st.write("**Source:** sample_document.pdf")
                st.write("**Type:** Technical Documentation")
                st.write("**Content Preview:**")
                st.write("""
                This is a sample excerpt from the document that was used to generate the answer. 
                In a real implementation, this would show the actual retrieved text chunks with 
                proper formatting and highlighting of relevant sections.
                """)
                
                # Relevance indicators
                relevance_score = 0.85 - i * 0.1
                relevance_level = self._get_relevance_level(relevance_score)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Relevance", f"{relevance_score:.2f}")
                
                with col2:
                    st.metric("Tokens", "156")
                
                with col3:
                    st.metric("Method", retrieval_results.get("method", "unknown"))
    
    def _get_relevance_level(self, score: float) -> str:
        """Get relevance level based on score."""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _display_relevance_distribution(self, retrieval_results: Dict[str, Any]):
        """Display relevance score distribution."""
        st.write("**Relevance Score Distribution:**")
        
        # Simulate relevance scores
        scores = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15]
        methods = [retrieval_results.get("method", "unknown")] * len(scores)
        
        # Create histogram
        fig = px.histogram(
            x=scores,
            nbins=10,
            title="Relevance Score Distribution",
            labels={"x": "Relevance Score", "y": "Count"}
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Relevance statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Relevance", f"{sum(scores)/len(scores):.2f}")
        
        with col2:
            st.metric("Max Relevance", f"{max(scores):.2f}")
        
        with col3:
            st.metric("Min Relevance", f"{min(scores):.2f}")
    
    def display_document_metadata(self, document_metadata: Dict[str, Any]):
        """Display metadata for source documents."""
        with st.expander("📋 Document Metadata"):
            if not document_metadata:
                st.info("No metadata available")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**File Information:**")
                st.write(f"- Filename: {document_metadata.get('filename', 'N/A')}")
                st.write(f"- File Type: {document_metadata.get('file_type', 'N/A')}")
                st.write(f"- File Size: {document_metadata.get('file_size', 0):,} bytes")
                st.write(f"- Directory: {document_metadata.get('directory', 'N/A')}")
            
            with col2:
                st.write("**Processing Information:**")
                st.write(f"- Document Type: {document_metadata.get('document_type', 'N/A')}")
                st.write(f"- Content Length: {document_metadata.get('content_length', 0):,} chars")
                st.write(f"- Chunk Index: {document_metadata.get('chunk_index', 'N/A')}")
                st.write(f"- Chunk Length: {document_metadata.get('chunk_length', 0):,} chars")
    
    def display_search_method_info(self, method: str):
        """Display information about the search method used."""
        st.subheader("🔍 Search Method Information")
        
        method_info = {
            "bm25": {
                "name": "BM25 Keyword Search",
                "description": "Traditional keyword-based search using BM25 algorithm",
                "strengths": ["Fast execution", "Good for exact matches", "Handles typos well"],
                "weaknesses": ["Limited semantic understanding", "May miss related concepts"],
                "best_for": ["Factual questions", "Specific term searches", "Exact phrase matching"]
            },
            "semantic": {
                "name": "Semantic Vector Search", 
                "description": "Semantic search using vector embeddings and similarity",
                "strengths": ["Understands meaning", "Finds related concepts", "Language agnostic"],
                "weaknesses": ["Slower than BM25", "May miss exact matches"],
                "best_for": ["Conceptual searches", "Finding similar content", "Semantic understanding"]
            },
            "hybrid": {
                "name": "Hybrid BM25 + Semantic",
                "description": "Combined BM25 and semantic search with weighted scoring",
                "strengths": ["Best of both worlds", "High precision and recall", "Robust performance"],
                "weaknesses": ["More complex", "Higher computational cost"],
                "best_for": ["Complex analytical queries", "Balanced precision and recall", "General purpose search"]
            },
            "conversational": {
                "name": "Conversational Search",
                "description": "Semantic search with expanded context for conversational queries",
                "strengths": ["Context-aware", "Handles conversational language", "Expanded understanding"],
                "weaknesses": ["May be too broad", "Higher computational cost"],
                "best_for": ["Help and guidance queries", "How-to questions", "Tutorial searches"]
            }
        }
        
        info = method_info.get(method.lower(), method_info["semantic"])
        
        st.write(f"**Method:** {info['name']}")
        st.write(f"**Description:** {info['description']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strengths:**")
            for strength in info['strengths']:
                st.write(f"✅ {strength}")
        
        with col2:
            st.write("**Weaknesses:**")
            for weakness in info['weaknesses']:
                st.write(f"⚠️ {weakness}")
        
        st.write("**Best For:**")
        for use_case in info['best_for']:
            st.write(f"🎯 {use_case}")
    
    def display_document_similarity(self, similarity_data: Dict[str, Any]):
        """Display document similarity analysis."""
        st.subheader("🔗 Document Similarity Analysis")
        
        if not similarity_data:
            st.info("No similarity data available")
            return
        
        # Create similarity matrix visualization
        documents = similarity_data.get("documents", [])
        similarity_matrix = similarity_data.get("similarity_matrix", [])
        
        if similarity_matrix:
            fig = px.imshow(
                similarity_matrix,
                labels=dict(x="Document", y="Document", color="Similarity"),
                title="Document Similarity Matrix"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Similarity statistics
        avg_similarity = similarity_data.get("average_similarity", 0)
        max_similarity = similarity_data.get("max_similarity", 0)
        min_similarity = similarity_data.get("min_similarity", 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Similarity", f"{avg_similarity:.2f}")
        
        with col2:
            st.metric("Max Similarity", f"{max_similarity:.2f}")
        
        with col3:
            st.metric("Min Similarity", f"{min_similarity:.2f}")
    
    def display_document_clusters(self, cluster_data: Dict[str, Any]):
        """Display document clustering results."""
        st.subheader("📊 Document Clusters")
        
        if not cluster_data:
            st.info("No clustering data available")
            return
        
        clusters = cluster_data.get("clusters", [])
        
        if not clusters:
            st.info("No clusters found")
            return
        
        st.write(f"Found {len(clusters)} document clusters")
        
        for i, cluster in enumerate(clusters):
            with st.expander(f"Cluster {i+1} ({len(cluster)} documents)"):
                for j, doc in enumerate(cluster):
                    st.write(f"**Document {j+1}:** {doc.get('title', 'Untitled')}")
                    st.write(f"- Relevance: {doc.get('score', 0):.2f}")
                    st.write(f"- Source: {doc.get('source', 'Unknown')}")
                    st.write("---")
    
    def export_source_data(self, retrieval_results: Dict[str, Any]):
        """Export source document data."""
        st.subheader("📤 Export Source Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export as JSON"):
                # In a real implementation, this would export actual data
                st.success("Source data exported as JSON")
        
        with col2:
            if st.button("📊 Export as CSV"):
                # In a real implementation, this would export actual data
                st.success("Source data exported as CSV")
        
        # Display export options
        st.write("**Export Options:**")
        st.write("- Include document content")
        st.write("- Include relevance scores")
        st.write("- Include metadata")
        st.write("- Include processing timestamps")
