"""
Context inspector component for Streamlit UI.
Displays context optimization results and token savings.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

class ContextInspector:
    """Component for inspecting context optimization results."""
    
    def __init__(self):
        self.optimization_colors = {
            "original": "#ff6b6b",
            "optimized": "#4ecdc4",
            "savings": "#45b7d1"
        }
    
    def display_optimization_results(self, optimization: Dict[str, Any]):
        """Display complete optimization results."""
        st.subheader("⚡ Context Optimization")
        
        # Main optimization metrics
        self._display_optimization_metrics(optimization)
        
        # Token savings visualization
        self._display_token_savings(optimization)
        
        # Detailed optimization stats
        self._display_detailed_stats(optimization)
    
    def _display_optimization_metrics(self, optimization: Dict[str, Any]):
        """Display main optimization metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        original_count = optimization.get("original_count", 0)
        optimized_count = optimization.get("optimized_count", 0)
        
        with col1:
            st.metric(
                "Original Results",
                original_count,
                delta=None
            )
        
        with col2:
            st.metric(
                "Optimized Results", 
                optimized_count,
                delta=optimized_count - original_count
            )
        
        with col3:
            if original_count > 0:
                reduction_percent = ((original_count - optimized_count) / original_count) * 100
                st.metric(
                    "Reduction",
                    f"{reduction_percent:.1f}%",
                    delta=f"-{original_count - optimized_count}"
                )
            else:
                st.metric("Reduction", "0%")
        
        with col4:
            enabled = optimization.get("enabled", False)
            status = "✅ Enabled" if enabled else "❌ Disabled"
            st.metric("Optimization", status)
    
    def _display_token_savings(self, optimization: Dict[str, Any]):
        """Display token savings visualization."""
        stats = optimization.get("stats", {})
        
        if not stats:
            st.info("No detailed optimization statistics available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Deduplication stats
            dedup_stats = stats.get("deduplication", {})
            if dedup_stats:
                self._display_deduplication_chart(dedup_stats)
        
        with col2:
            # Compression stats
            comp_stats = stats.get("compression", {})
            if comp_stats:
                self._display_compression_chart(comp_stats)
    
    def _display_deduplication_chart(self, dedup_stats: Dict[str, Any]):
        """Display deduplication chart."""
        st.write("**🔄 Deduplication Results**")
        
        original_tokens = dedup_stats.get("original_tokens", 0)
        deduplicated_tokens = dedup_stats.get("deduplicated_tokens", 0)
        token_savings = dedup_stats.get("token_savings", 0)
        
        # Create pie chart
        labels = ["Removed", "Kept"]
        values = [token_savings, deduplicated_tokens]
        colors = ["#ff6b6b", "#4ecdc4"]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title="Token Distribution",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        st.write(f"**Token Savings:** {token_savings:,}")
        st.write(f"**Reduction:** {dedup_stats.get('token_reduction_percent', 0):.1f}%")
    
    def _display_compression_chart(self, comp_stats: Dict[str, Any]):
        """Display compression chart."""
        st.write("**🗜️ Compression Results**")
        
        original_tokens = comp_stats.get("original_tokens", 0)
        compressed_tokens = comp_stats.get("compressed_tokens", 0)
        compression_ratio = comp_stats.get("compression_ratio", 1.0)
        
        # Create bar chart
        categories = ["Original", "Compressed"]
        values = [original_tokens, compressed_tokens]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=["#ff6b6b", "#4ecdc4"],
                text=[f"{v:,}" for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Token Count Comparison",
            yaxis_title="Tokens",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        st.write(f"**Compression Ratio:** {compression_ratio:.2f}")
        st.write(f"**Compression:** {comp_stats.get('compression_percent', 0):.1f}%")
    
    def _display_detailed_stats(self, optimization: Dict[str, Any]):
        """Display detailed optimization statistics."""
        with st.expander("📊 Detailed Optimization Statistics"):
            stats = optimization.get("stats", {})
            
            if not stats:
                st.info("No detailed statistics available")
                return
            
            # Overall reduction
            overall_reduction = stats.get("overall_reduction_percent", 0)
            st.write(f"**Overall Reduction:** {overall_reduction:.1f}%")
            
            # Deduplication details
            dedup_stats = stats.get("deduplication", {})
            if dedup_stats:
                st.write("**Deduplication Details:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"- Original Count: {dedup_stats.get('original_count', 0)}")
                    st.write(f"- Deduplicated Count: {dedup_stats.get('deduplicated_count', 0)}")
                    st.write(f"- Removed Count: {dedup_stats.get('removed_count', 0)}")
                
                with col2:
                    st.write(f"- Original Tokens: {dedup_stats.get('original_tokens', 0):,}")
                    st.write(f"- Deduplicated Tokens: {dedup_stats.get('deduplicated_tokens', 0):,}")
                    st.write(f"- Token Savings: {dedup_stats.get('token_savings', 0):,}")
            
            # Compression details
            comp_stats = stats.get("compression", {})
            if comp_stats:
                st.write("**Compression Details:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"- Original Tokens: {comp_stats.get('original_tokens', 0):,}")
                    st.write(f"- Compressed Tokens: {comp_stats.get('compressed_tokens', 0):,}")
                    st.write(f"- Token Savings: {comp_stats.get('token_savings', 0):,}")
                
                with col2:
                    st.write(f"- Compression Ratio: {comp_stats.get('compression_ratio', 1):.2f}")
                    st.write(f"- Compression %: {comp_stats.get('compression_percent', 0):.1f}%")
                    st.write(f"- Max Tokens: {comp_stats.get('max_tokens', 0):,}")
    
    def display_optimization_timeline(self, optimization_history: List[Dict[str, Any]]):
        """Display optimization performance over time."""
        if not optimization_history:
            st.info("No optimization history available")
            return
        
        st.subheader("📈 Optimization Performance Timeline")
        
        # Prepare data
        timestamps = [item.get("timestamp", "") for item in optimization_history]
        token_reductions = [item.get("token_reduction_percent", 0) for item in optimization_history]
        processing_times = [item.get("processing_time", 0) for item in optimization_history]
        
        # Create timeline chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=token_reductions,
            mode='lines+markers',
            name='Token Reduction %',
            line=dict(color='#4ecdc4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Token Reduction Over Time",
            xaxis_title="Time",
            yaxis_title="Token Reduction (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Processing time chart
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=timestamps,
            y=processing_times,
            mode='lines+markers',
            name='Processing Time (s)',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8)
        ))
        
        fig2.update_layout(
            title="Processing Time Over Time",
            xaxis_title="Time",
            yaxis_title="Processing Time (seconds)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def display_optimization_comparison(self, comparison_data: Dict[str, Any]):
        """Display comparison of different optimization methods."""
        st.subheader("⚖️ Optimization Method Comparison")
        
        method_comparison = comparison_data.get("method_comparison", {})
        
        if not method_comparison:
            st.info("No comparison data available")
            return
        
        # Create comparison table
        comparison_rows = []
        for method, stats in method_comparison.items():
            comparison_rows.append({
                "Method": method.title(),
                "Optimizations": stats.get("total_optimizations", 0),
                "Avg Token Reduction %": f"{stats.get('avg_token_reduction_percent', 0):.1f}",
                "Avg Processing Time": f"{stats.get('avg_processing_time', 0):.2f}s",
                "Total Token Savings": f"{stats.get('total_token_savings', 0):,}",
                "Efficiency Score": f"{stats.get('efficiency_score', 0):.2f}"
            })
        
        # Sort by efficiency score
        comparison_rows.sort(key=lambda x: float(x["Efficiency Score"]), reverse=True)
        
        st.dataframe(comparison_rows, use_container_width=True)
        
        # Highlight best method
        best_method = comparison_data.get("best_method")
        if best_method:
            st.success(f"🏆 Best performing method: **{best_method.title()}**")
    
    def display_token_usage_analysis(self, token_analysis: Dict[str, Any]):
        """Display token usage analysis."""
        st.subheader("🔍 Token Usage Analysis")
        
        if "error" in token_analysis:
            st.error(f"Error in token analysis: {token_analysis['error']}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic metrics
            st.write("**Basic Metrics:**")
            st.write(f"- Total Results: {token_analysis.get('total_results', 0)}")
            st.write(f"- Total Tokens: {token_analysis.get('total_tokens', 0):,}")
            st.write(f"- Avg Tokens per Result: {token_analysis.get('avg_tokens_per_result', 0):.1f}")
            st.write(f"- Token Efficiency: {token_analysis.get('token_efficiency', 0):.1f}")
        
        with col2:
            # Distribution
            st.write("**Length Distribution:**")
            distribution = token_analysis.get("length_distribution", {})
            st.write(f"- Short (<50 tokens): {distribution.get('short', 0)}")
            st.write(f"- Medium (50-200 tokens): {distribution.get('medium', 0)}")
            st.write(f"- Long (>200 tokens): {distribution.get('long', 0)}")
        
        # Percentiles chart
        percentiles = {
            "Min": token_analysis.get("min_tokens", 0),
            "25th": token_analysis.get("median_tokens", 0) * 0.75,  # Approximate
            "50th": token_analysis.get("median_tokens", 0),
            "75th": token_analysis.get("p75_tokens", 0),
            "90th": token_analysis.get("p90_tokens", 0),
            "Max": token_analysis.get("max_tokens", 0)
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(percentiles.keys()),
                y=list(percentiles.values()),
                marker_color="#45b7d1"
            )
        ])
        
        fig.update_layout(
            title="Token Count Percentiles",
            xaxis_title="Percentile",
            yaxis_title="Tokens",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
