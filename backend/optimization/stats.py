"""
Statistics and metrics for context optimization.
Tracks token savings, compression ratios, and optimization effectiveness.
"""

from typing import List, Dict, Any, Optional
import time
from datetime import datetime
from ..core.logger import app_logger
from ..core.utils import count_tokens

class OptimizationStats:
    """Tracks and analyzes optimization statistics."""
    
    def __init__(self):
        self.logger = app_logger
        self.stats_history = []
    
    def record_optimization(self, original_results: List[Dict[str, Any]], 
                          optimized_results: List[Dict[str, Any]],
                          optimization_type: str,
                          processing_time: float = 0.0) -> Dict[str, Any]:
        """
        Record optimization statistics.
        
        Args:
            original_results: Results before optimization
            optimized_results: Results after optimization
            optimization_type: Type of optimization (deduplication, compression, etc.)
            processing_time: Time taken for optimization
            
        Returns:
            Statistics dictionary
        """
        try:
            # Calculate basic stats
            original_count = len(original_results)
            optimized_count = len(optimized_results)
            
            original_tokens = sum(count_tokens(result["content"]) for result in original_results)
            optimized_tokens = sum(count_tokens(result["content"]) for result in optimized_results)
            
            # Calculate metrics
            token_savings = original_tokens - optimized_tokens
            token_reduction_percent = (token_savings / original_tokens * 100) if original_tokens > 0 else 0
            count_reduction_percent = ((original_count - optimized_count) / original_count * 100) if original_count > 0 else 0
            
            # Create stats record
            stats_record = {
                "timestamp": datetime.now().isoformat(),
                "optimization_type": optimization_type,
                "processing_time": processing_time,
                "original_count": original_count,
                "optimized_count": optimized_count,
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "token_savings": token_savings,
                "token_reduction_percent": token_reduction_percent,
                "count_reduction_percent": count_reduction_percent,
                "compression_ratio": optimized_tokens / original_tokens if original_tokens > 0 else 1.0
            }
            
            # Store in history
            self.stats_history.append(stats_record)
            
            self.logger.info(f"Optimization recorded: {optimization_type} - {token_reduction_percent:.1f}% token reduction")
            
            return stats_record
            
        except Exception as e:
            self.logger.error(f"Error recording optimization stats: {str(e)}")
            return {"error": str(e)}
    
    def get_optimization_summary(self, optimization_type: str = None, 
                               limit: int = 10) -> Dict[str, Any]:
        """Get summary statistics for optimizations."""
        try:
            # Filter by optimization type if specified
            filtered_stats = self.stats_history
            if optimization_type:
                filtered_stats = [s for s in self.stats_history if s["optimization_type"] == optimization_type]
            
            if not filtered_stats:
                return {"message": "No optimization statistics available"}
            
            # Get recent stats
            recent_stats = filtered_stats[-limit:] if limit else filtered_stats
            
            # Calculate averages
            avg_token_reduction = sum(s["token_reduction_percent"] for s in recent_stats) / len(recent_stats)
            avg_count_reduction = sum(s["count_reduction_percent"] for s in recent_stats) / len(recent_stats)
            avg_processing_time = sum(s["processing_time"] for s in recent_stats) / len(recent_stats)
            avg_compression_ratio = sum(s["compression_ratio"] for s in recent_stats) / len(recent_stats)
            
            # Calculate totals
            total_token_savings = sum(s["token_savings"] for s in recent_stats)
            total_processing_time = sum(s["processing_time"] for s in recent_stats)
            
            return {
                "optimization_type": optimization_type or "all",
                "total_optimizations": len(recent_stats),
                "avg_token_reduction_percent": avg_token_reduction,
                "avg_count_reduction_percent": avg_count_reduction,
                "avg_processing_time": avg_processing_time,
                "avg_compression_ratio": avg_compression_ratio,
                "total_token_savings": total_token_savings,
                "total_processing_time": total_processing_time,
                "recent_stats": recent_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting optimization summary: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_trends(self, optimization_type: str = None, 
                            days: int = 7) -> Dict[str, Any]:
        """Get performance trends over time."""
        try:
            # Filter by optimization type and time range
            filtered_stats = self.stats_history
            if optimization_type:
                filtered_stats = [s for s in self.stats_history if s["optimization_type"] == optimization_type]
            
            # Group by day
            daily_stats = {}
            for stat in filtered_stats:
                date = stat["timestamp"][:10]  # Extract date part
                if date not in daily_stats:
                    daily_stats[date] = []
                daily_stats[date].append(stat)
            
            # Calculate daily averages
            daily_trends = []
            for date, day_stats in sorted(daily_stats.items()):
                avg_token_reduction = sum(s["token_reduction_percent"] for s in day_stats) / len(day_stats)
                avg_count_reduction = sum(s["count_reduction_percent"] for s in day_stats) / len(day_stats)
                total_token_savings = sum(s["token_savings"] for s in day_stats)
                
                daily_trends.append({
                    "date": date,
                    "optimizations": len(day_stats),
                    "avg_token_reduction_percent": avg_token_reduction,
                    "avg_count_reduction_percent": avg_count_reduction,
                    "total_token_savings": total_token_savings
                })
            
            return {
                "optimization_type": optimization_type or "all",
                "trend_period_days": days,
                "daily_trends": daily_trends
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance trends: {str(e)}")
            return {"error": str(e)}
    
    def compare_optimization_methods(self) -> Dict[str, Any]:
        """Compare different optimization methods."""
        try:
            # Group stats by optimization type
            method_stats = {}
            for stat in self.stats_history:
                opt_type = stat["optimization_type"]
                if opt_type not in method_stats:
                    method_stats[opt_type] = []
                method_stats[opt_type].append(stat)
            
            # Calculate comparison metrics
            comparison = {}
            for method, stats in method_stats.items():
                if not stats:
                    continue
                
                avg_token_reduction = sum(s["token_reduction_percent"] for s in stats) / len(stats)
                avg_count_reduction = sum(s["count_reduction_percent"] for s in stats) / len(stats)
                avg_processing_time = sum(s["processing_time"] for s in stats) / len(stats)
                total_token_savings = sum(s["token_savings"] for s in stats)
                
                comparison[method] = {
                    "total_optimizations": len(stats),
                    "avg_token_reduction_percent": avg_token_reduction,
                    "avg_count_reduction_percent": avg_count_reduction,
                    "avg_processing_time": avg_processing_time,
                    "total_token_savings": total_token_savings,
                    "efficiency_score": avg_token_reduction / avg_processing_time if avg_processing_time > 0 else 0
                }
            
            return {
                "method_comparison": comparison,
                "best_method": max(comparison.items(), key=lambda x: x[1]["efficiency_score"])[0] if comparison else None
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing optimization methods: {str(e)}")
            return {"error": str(e)}
    
    def get_token_usage_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze token usage patterns in results."""
        try:
            if not results:
                return {"error": "No results provided"}
            
            # Calculate token statistics
            token_counts = [count_tokens(result["content"]) for result in results]
            
            total_tokens = sum(token_counts)
            avg_tokens = total_tokens / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            
            # Calculate percentiles
            sorted_tokens = sorted(token_counts)
            p50 = sorted_tokens[len(sorted_tokens) // 2]
            p75 = sorted_tokens[int(len(sorted_tokens) * 0.75)]
            p90 = sorted_tokens[int(len(sorted_tokens) * 0.90)]
            
            # Analyze content length distribution
            length_categories = {
                "short": len([t for t in token_counts if t < 50]),
                "medium": len([t for t in token_counts if 50 <= t < 200]),
                "long": len([t for t in token_counts if t >= 200])
            }
            
            return {
                "total_results": len(results),
                "total_tokens": total_tokens,
                "avg_tokens_per_result": avg_tokens,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "median_tokens": p50,
                "p75_tokens": p75,
                "p90_tokens": p90,
                "length_distribution": length_categories,
                "token_efficiency": total_tokens / len(results) if results else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing token usage: {str(e)}")
            return {"error": str(e)}
    
    def clear_history(self):
        """Clear optimization statistics history."""
        self.stats_history.clear()
        self.logger.info("Optimization statistics history cleared")
    
    def export_stats(self, format: str = "json") -> Dict[str, Any]:
        """Export statistics in various formats."""
        try:
            if format == "json":
                return {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_records": len(self.stats_history),
                    "stats": self.stats_history
                }
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            self.logger.error(f"Error exporting stats: {str(e)}")
            return {"error": str(e)}
