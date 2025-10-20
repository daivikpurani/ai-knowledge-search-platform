"""
Query interface component for Streamlit UI.
Handles query input and displays search results.
"""

import streamlit as st
import requests
from typing import Dict, Any, List

class QueryInterface:
    """Query interface component for the Streamlit UI."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000/api/v1"):
        self.api_base_url = api_base_url
    
    def display_query_form(self) -> tuple:
        """Display query input form and return query and search button state."""
        st.subheader("🔍 Search Query")
        
        # Query input with examples
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is machine learning? How does neural network work?",
            help="Ask any question about your documents. The system will intelligently route your query and provide an optimized response."
        )
        
        # Example queries
        with st.expander("💡 Example Queries"):
            examples = [
                "What is artificial intelligence?",
                "How do neural networks work?",
                "Explain machine learning algorithms",
                "What are the benefits of deep learning?",
                "Compare supervised vs unsupervised learning"
            ]
            
            for example in examples:
                if st.button(f"📝 {example}", key=f"example_{example}"):
                    st.session_state.query_input = example
                    st.rerun()
        
        # Search controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear")
        
        if clear_button:
            st.session_state.query_input = ""
            st.rerun()
        
        # Use session state for query input
        if hasattr(st.session_state, 'query_input'):
            query = st.session_state.query_input
        
        return query, search_button
    
    def display_search_results(self, response: Dict[str, Any]):
        """Display search results and answer."""
        st.subheader("📝 Answer")
        
        # Display LLM response
        llm_response = response.get("llm_response", {})
        answer = llm_response.get("text", "No answer generated")
        
        # Display answer in a nice format
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #007bff;">
            {answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Display response metadata
        with st.expander("📊 Response Details"):
            self._display_response_metadata(llm_response, response)
    
    def _display_response_metadata(self, llm_response: Dict[str, Any], full_response: Dict[str, Any]):
        """Display response metadata."""
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
            
            st.write("**Processing Time:**", f"{full_response.get('processing_time', 0):.2f}s")
    
    def display_source_documents(self, retrieval_results: Dict[str, Any]):
        """Display source documents used for the answer."""
        st.subheader("📚 Source Documents")
        
        count = retrieval_results.get("count", 0)
        method = retrieval_results.get("method", "unknown")
        
        if count > 0:
            st.info(f"Found {count} relevant documents using {method} search")
            
            # Placeholder for actual document display
            # In a full implementation, this would show the actual retrieved documents
            st.write("Source documents would be displayed here with:")
            st.write("- Document titles and sources")
            st.write("- Relevance scores")
            st.write("- Excerpts from each document")
            st.write("- Links to full documents")
        else:
            st.warning("No relevant documents found for this query")
    
    def make_search_request(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Make search request to the API."""
        try:
            request_data = {
                "query": query,
                "max_results": max_results
            }
            
            response = requests.post(
                f"{self.api_base_url}/query/rag",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def display_error(self, error_message: str):
        """Display error message."""
        st.error(f"❌ Error: {error_message}")
        
        # Provide troubleshooting tips
        with st.expander("🔧 Troubleshooting Tips"):
            st.write("1. **Check API Connection:** Make sure the backend server is running on port 8000")
            st.write("2. **Verify Query Format:** Ensure your query is between 3-1000 characters")
            st.write("3. **Check Documents:** Make sure documents have been ingested into the system")
            st.write("4. **Review Logs:** Check the backend logs for detailed error information")
    
    def display_loading_state(self):
        """Display loading state during search."""
        with st.spinner("Processing your query..."):
            st.info("🤖 The AI is analyzing your query and retrieving relevant information...")
            st.info("🎯 Determining the best search strategy...")
            st.info("⚡ Optimizing context for better results...")
            st.info("📝 Generating response...")
    
    def display_success_message(self, processing_time: float):
        """Display success message with processing time."""
        st.success(f"✅ Query processed successfully in {processing_time:.2f} seconds!")
        
        # Show performance indicator
        if processing_time < 2.0:
            st.info("🚀 Excellent performance!")
        elif processing_time < 5.0:
            st.info("⚡ Good performance")
        else:
            st.info("⏳ Processing took longer than usual")
