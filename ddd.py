# ============================================
# performance/optimizer.py - Performance Optimization Suite
# ============================================

"""
Performance optimization tools for Neo4j Enhanced Agent
Includes query optimization, caching, and performance monitoring
"""

import asyncio
import time
import statistics
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import psutil
import aioredis
from functools import wraps
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("performance_optimizer")

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query: str
    execution_time: float
    timestamp: float
    success: bool
    result_count: int
    tool_used: str
    session_id: str
    error: Optional[str] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_connections: int
    neo4j_status: str
    response_times: List[float] = field(default_factory=list)

class QueryCache:
    """Advanced query caching system"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.redis_client: Optional[aioredis.Redis] = None
    
    async def init_redis(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis connection for distributed caching"""
        try:
            self.redis_client = aioredis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("✅ Redis cache initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using in-memory cache: {e}")
    
    def _generate_cache_key(self, query: str, params: Dict = None) -> str:
        """Generate cache key for query"""
        key_data = f"{query}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _is_cacheable_query(self, query: str) -> bool:
        """Determine if query is cacheable"""
        query_upper = query.upper().strip()
        
        # Only cache read operations
        if not query_upper.startswith(("MATCH", "RETURN", "CALL", "SHOW")):
            return False
        
        # Don't cache queries with time-dependent functions
        time_functions = ["DATETIME()", "DATE()", "TIME()", "TIMESTAMP()", "RAND()"]
        if any(func in query_upper for func in time_functions):
            return False
        
        return True
    
    async def get(self, query: str, params: Dict = None) -> Optional[Any]:
        """Get cached result"""
        if not self._is_cacheable_query(query):
            return None
        
        cache_key = self._generate_cache_key(query, params)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"query_cache:{cache_key}")
                if cached_data:
                    result = json.loads(cached_data)
                    self.access_count[cache_key] += 1
                    logger.debug(f"📋 Cache HIT (Redis): {cache_key[:8]}")
                    return result
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}")
        
        # Fall back to in-memory cache
        if cache_key in self.cache:
            # Check TTL
            if time.time() - self.timestamps[cache_key] < self.ttl_seconds:
                self.access_count[cache_key] += 1
                logger.debug(f"📋 Cache HIT (Memory): {cache_key[:8]}")
                return self.cache[cache_key]
            else:
                # Expired
                del self.cache[cache_key]
                del self.timestamps[cache_key]
        
        logger.debug(f"📋 Cache MISS: {cache_key[:8]}")
        return None
    
    async def set(self, query: str, params: Dict, result: Any):
        """Cache query result"""
        if not self._is_cacheable_query(query):
            return
        
        cache_key = self._generate_cache_key(query, params)
        current_time = time.time()
        
        # Store in Redis
        if self.redis_client:
            try:
                cached_data = json.dumps(result, default=str)
                await self.redis_client.setex(
                    f"query_cache:{cache_key}",
                    self.ttl_seconds,
                    cached_data
                )
                logger.debug(f"📋 Cached to Redis: {cache_key[:8]}")
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Store in memory cache
        self.cache[cache_key] = result
        self.timestamps[cache_key] = current_time
        
        # Cleanup if cache is too large
        if len(self.cache) > self.max_size:
            # Remove least recently used items
            sorted_keys = sorted(
                self.access_count.keys(),
                key=lambda k: (self.access_count[k], self.timestamps.get(k, 0))
            )
            
            keys_to_remove = sorted_keys[:len(sorted_keys) - self.max_size + 100]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.timestamps.pop(key, None)
                self.access_count.pop(key, None)
        
        logger.debug(f"📋 Cached to Memory: {cache_key[:8]}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = sum(self.access_count.values())
        
        return {
            "cache_size": len(self.cache),
            "total_accesses": total_accesses,
            "redis_enabled": self.redis_client is not None,
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size
        }

class PerformanceMonitor:
    """Comprehensive performance monitoring"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.query_metrics: deque = deque(maxlen=window_size)
        self.system_metrics: deque = deque(maxlen=window_size)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.slow_queries: List[QueryMetrics] = []
        self.slow_query_threshold = 5.0  # seconds
    
    def record_query(self, metrics: QueryMetrics):
        """Record query performance metrics"""
        self.query_metrics.append(metrics)
        
        if not metrics.success:
            self.error_counts[metrics.error or "Unknown"] += 1
        
        if metrics.execution_time > self.slow_query_threshold:
            self.slow_queries.append(metrics)
            # Keep only last 100 slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
        
        logger.debug(f"📊 Query recorded: {metrics.execution_time:.2f}s - {metrics.query[:50]}...")
    
    def record_system(self, metrics: SystemMetrics):
        """Record system performance metrics"""
        self.system_metrics.append(metrics)
        logger.debug(f"💻 System metrics: CPU {metrics.cpu_percent:.1f}%, RAM {metrics.memory_percent:.1f}%")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.query_metrics:
            return {"error": "No metrics available"}
        
        query_times = [m.execution_time for m in self.query_metrics if m.success]
        
        if not query_times:
            return {"error": "No successful queries recorded"}
        
        # Query statistics
        query_stats = {
            "total_queries": len(self.query_metrics),
            "successful_queries": len(query_times),
            "failed_queries": len(self.query_metrics) - len(query_times),
            "success_rate": len(query_times) / len(self.query_metrics) * 100,
            "avg_response_time": statistics.mean(query_times),
            "median_response_time": statistics.median(query_times),
            "min_response_time": min(query_times),
            "max_response_time": max(query_times),
            "std_dev": statistics.stdev(query_times) if len(query_times) > 1 else 0,
            "slow_queries_count": len(self.slow_queries)
        }
        
        # Tool usage statistics
        tool_usage = defaultdict(int)
        for metric in self.query_metrics:
            tool_usage[metric.tool_used] += 1
        
        # System statistics
        system_stats = {}
        if self.system_metrics:
            recent_system = list(self.system_metrics)[-10:]  # Last 10 readings
            system_stats = {
                "avg_cpu_percent": statistics.mean(m.cpu_percent for m in recent_system),
                "avg_memory_percent": statistics.mean(m.memory_percent for m in recent_system),
                "avg_memory_used_mb": statistics.mean(m.memory_used_mb for m in recent_system),
                "avg_active_connections": statistics.mean(m.active_connections for m in recent_system)
            }
        
        return {
            "timestamp": time.time(),
            "query_performance": query_stats,
            "tool_usage": dict(tool_usage),
            "system_performance": system_stats,
            "error_summary": dict(self.error_counts),
            "top_slow_queries": [
                {
                    "query": q.query[:100],
                    "time": q.execution_time,
                    "tool": q.tool_used,
                    "timestamp": q.timestamp
                }
                for q in sorted(self.slow_queries, key=lambda x: x.execution_time, reverse=True)[:10]
            ]
        }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        recent_queries = list(self.query_metrics)[-100:]  # Last 100 queries
        
        if not recent_queries:
            return {"status": "no_data"}
        
        successful_recent = [q for q in recent_queries if q.success]
        
        return {
            "timestamp": time.time(),
            "queries_per_minute": len(recent_queries),
            "success_rate": len(successful_recent) / len(recent_queries) * 100 if recent_queries else 0,
            "avg_response_time": statistics.mean(q.execution_time for q in successful_recent) if successful_recent else 0,
            "active_sessions": len(set(q.session_id for q in recent_queries)),
            "errors_last_hour": sum(1 for q in recent_queries if not q.success),
            "system_health": "healthy" if self.system_metrics and self.system_metrics[-1].cpu_percent < 80 else "warning"
        }

class QueryOptimizer:
    """Automatic query optimization"""
    
    def __init__(self):
        self.optimization_rules = [
            self._optimize_count_queries,
            self._optimize_limit_queries,
            self._optimize_index_hints,
            self._optimize_relationship_patterns
        ]
    
    def _optimize_count_queries(self, query: str) -> str:
        """Optimize count queries"""
        query_upper = query.upper().strip()
        
        # Convert slow count(*) to faster count(n)
        if "COUNT(*)" in query_upper and "MATCH (N)" in query_upper:
            optimized = query.replace("COUNT(*)", "count(n)")
            logger.debug(f"🔧 Optimized count query: COUNT(*) → count(n)")
            return optimized
        
        return query
    
    def _optimize_limit_queries(self, query: str) -> str:
        """Add LIMIT to potentially large result queries"""
        query_upper = query.upper().strip()
        
        # Add LIMIT if missing and query could return many results
        if (query_upper.startswith("MATCH") and 
            "RETURN" in query_upper and 
            "LIMIT" not in query_upper and
            "COUNT" not in query_upper):
            
            optimized = query + " LIMIT 1000"
            logger.debug(f"🔧 Added safety LIMIT to query")
            return optimized
        
        return query
    
    def _optimize_index_hints(self, query: str) -> str:
        """Add index hints for common patterns"""
        # This would add index hints based on common patterns
        # For now, just return the original query
        return query
    
    def _optimize_relationship_patterns(self, query: str) -> str:
        """Optimize relationship traversal patterns"""
        # Optimize common relationship patterns
        # This could include using shortcut paths, etc.
        return query
    
    def optimize_query(self, query: str) -> tuple[str, List[str]]:
        """Apply optimization rules to query"""
        optimized_query = query
        applied_optimizations = []
        
        for rule in self.optimization_rules:
            old_query = optimized_query
            optimized_query = rule(optimized_query)
            
            if old_query != optimized_query:
                applied_optimizations.append(rule.__name__)
        
        return optimized_query, applied_optimizations

class PerformanceDecorator:
    """Performance monitoring decorators"""
    
    def __init__(self, monitor: PerformanceMonitor, cache: QueryCache = None):
        self.monitor = monitor
        self.cache = cache
    
    def monitor_performance(self, tool_name: str = "unknown"):
        """Decorator to monitor function performance"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error = None
                result = None
                
                try:
                    # Check cache first if available
                    if self.cache and len(args) > 0:
                        cache_result = await self.cache.get(str(args[0]), kwargs)
                        if cache_result is not None:
                            return cache_result
                    
                    result = await func(*args, **kwargs)
                    
                    # Cache successful results
                    if self.cache and result and len(args) > 0:
                        await self.cache.set(str(args[0]), kwargs, result)
                    
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    execution_time = time.time() - start_time
                    
                    # Record metrics
                    metrics = QueryMetrics(
                        query=str(args[0]) if args else "unknown",
                        execution_time=execution_time,
                        timestamp=time.time(),
                        success=success,
                        result_count=len(result) if isinstance(result, (list, dict)) else 1,
                        tool_used=tool_name,
                        session_id=kwargs.get("session_id", "unknown"),
                        error=error
                    )
                    
                    self.monitor.record_query(metrics)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error = None
                result = None
                
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    execution_time = time.time() - start_time
                    
                    metrics = QueryMetrics(
                        query=str(args[0]) if args else "unknown",
                        execution_time=execution_time,
                        timestamp=time.time(),
                        success=success,
                        result_count=len(result) if isinstance(result, (list, dict)) else 1,
                        tool_used=tool_name,
                        session_id=kwargs.get("session_id", "unknown"),
                        error=error
                    )
                    
                    self.monitor.record_query(metrics)
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.monitoring_active = False
        self.monitor_task = None
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous system monitoring"""
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"📊 Started system monitoring (interval: {interval}s)")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("📊 Stopped system monitoring")
    
    async def _monitor_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_system_metrics()
                self.monitor.record_system(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
            except Exception as e:
                logger.error(f"❌ System monitoring error: {e}")
            
            await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network connections (approximation of active connections)
        connections = len(psutil.net_connections())
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            disk_usage_percent=disk.percent,
            active_connections=connections,
            neo4j_status="unknown"  # Would be updated by health checks
        )
    
    async def _check_alerts(self, metrics: SystemMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.cpu_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > 90:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > 95:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        for alert in alerts:
            logger.warning(f"⚠️ PERFORMANCE ALERT: {alert}")

class PerformanceAnalyzer:
    """Advanced performance analysis"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query performance patterns"""
        if not self.monitor.query_metrics:
            return {"error": "No query data available"}
        
        queries = list(self.monitor.query_metrics)
        
        # Group queries by pattern
        patterns = defaultdict(list)
        for query in queries:
            # Simple pattern extraction (first few words)
            pattern = " ".join(query.query.split()[:3]).upper()
            patterns[pattern].append(query.execution_time)
        
        # Analyze each pattern
        pattern_analysis = {}
        for pattern, times in patterns.items():
            if len(times) > 1:
                pattern_analysis[pattern] = {
                    "count": len(times),
                    "avg_time": statistics.mean(times),
                    "median_time": statistics.median(times),
                    "max_time": max(times),
                    "std_dev": statistics.stdev(times),
                    "improvement_opportunity": max(times) > statistics.mean(times) * 2
                }
        
        return {
            "total_patterns": len(pattern_analysis),
            "patterns": dict(sorted(
                pattern_analysis.items(),
                key=lambda x: x[1]["avg_time"],
                reverse=True
            )[:20])  # Top 20 slowest patterns
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if not self.monitor.query_metrics:
            return ["Insufficient data for recommendations"]
        
        queries = list(self.monitor.query_metrics)
        successful_queries = [q for q in queries if q.success]
        
        if not successful_queries:
            return ["No successful queries to analyze"]
        
        # Check average response time
        avg_time = statistics.mean(q.execution_time for q in successful_queries)
        if avg_time > 2.0:
            recommendations.append(f"Average query time is high ({avg_time:.2f}s). Consider query optimization.")
        
        # Check error rate
        error_rate = (len(queries) - len(successful_queries)) / len(queries) * 100
        if error_rate > 5:
            recommendations.append(f"High error rate ({error_rate:.1f}%). Check query validation and error handling.")
        
        # Check slow queries
        slow_queries = len(self.monitor.slow_queries)
        if slow_queries > 10:
            recommendations.append(f"Many slow queries detected ({slow_queries}). Review query performance.")
        
        # Check cache effectiveness
        cache_stats = getattr(self.monitor, 'cache_stats', {})
        if cache_stats.get('hit_rate', 0) < 30:
            recommendations.append("Low cache hit rate. Consider expanding caching strategy.")
        
        if not recommendations:
            recommendations.append("Performance looks good! No immediate optimizations needed.")
        
        return recommendations

# ============================================
# Integration with Enhanced FastMCP Server
# ============================================

class EnhancedPerformanceIntegration:
    """Integration layer for performance monitoring with FastMCP server"""
    
    def __init__(self):
        self.cache = QueryCache(max_size=2000, ttl_seconds=600)
        self.monitor = PerformanceMonitor(window_size=2000)
        self.optimizer = QueryOptimizer()
        self.decorator = PerformanceDecorator(self.monitor, self.cache)
        self.analyzer = PerformanceAnalyzer(self.monitor)
        self.system_monitor = SystemMonitor(self.monitor)
    
    async def initialize(self, redis_url: str = None):
        """Initialize performance system"""
        if redis_url:
            await self.cache.init_redis(redis_url)
        
        await self.system_monitor.start_monitoring()
        logger.info("🚀 Performance monitoring system initialized")
    
    async def shutdown(self):
        """Shutdown performance system"""
        await self.system_monitor.stop_monitoring()
        logger.info("🛑 Performance monitoring system shutdown")
    
    def get_monitoring_endpoints(self):
        """Get additional FastAPI endpoints for monitoring"""
        from fastapi import APIRouter
        
        router = APIRouter(prefix="/performance", tags=["performance"])
        
        @router.get("/report")
        async def get_performance_report():
            """Get comprehensive performance report"""
            return self.monitor.get_performance_report()
        
        @router.get("/realtime")
        async def get_realtime_metrics():
            """Get real-time performance metrics"""
            return self.monitor.get_real_time_metrics()
        
        @router.get("/cache-stats")
        async def get_cache_stats():
            """Get cache statistics"""
            return self.cache.get_stats()
        
        @router.get("/query-patterns")
        async def get_query_patterns():
            """Analyze query patterns"""
            return self.analyzer.analyze_query_patterns()
        
        @router.get("/recommendations")
        async def get_recommendations():
            """Get optimization recommendations"""
            return {"recommendations": self.analyzer.get_optimization_recommendations()}
        
        @router.post("/clear-cache")
        async def clear_cache():
            """Clear query cache"""
            self.cache.cache.clear()
            self.cache.timestamps.clear()
            self.cache.access_count.clear()
            
            if self.cache.redis_client:
                await self.cache.redis_client.flushdb()
            
            return {"status": "cache_cleared"}
        
        return router

# Example usage:
"""
# In your enhanced_fastmcp_server.py, integrate like this:

from performance.optimizer import EnhancedPerformanceIntegration

# Initialize performance system
performance = EnhancedPerformanceIntegration()

@app.on_event("startup")
async def startup_event():
    # ... existing startup code ...
    await performance.initialize(redis_url="redis://localhost:6379")

@app.on_event("shutdown") 
async def shutdown_event():
    # ... existing shutdown code ...
    await performance.shutdown()

# Add performance monitoring endpoints
app.include_router(performance.get_monitoring_endpoints())

# Decorate your MCP tools with performance monitoring
@performance.decorator.monitor_performance("read_neo4j_cypher")
@mcp.tool()
async def read_neo4j_cypher(query: str, params: dict = {}):
    # ... existing implementation ...
    
@performance.decorator.monitor_performance("write_neo4j_cypher")
@mcp.tool()
async def write_neo4j_cypher(query: str, params: dict = {}):
    # ... existing implementation ...

# Use query optimization
optimized_query, optimizations = performance.optimizer.optimize_query(original_query)
"""

if __name__ == "__main__":
    # Example standalone usage
    async def main():
        performance = EnhancedPerformanceIntegration()
        await performance.initialize()
        
        # Simulate some metrics
        test_metrics = QueryMetrics(
            query="MATCH (n) RETURN count(n)",
            execution_time=1.5,
            timestamp=time.time(),
            success=True,
            result_count=1,
            tool_used="read_neo4j_cypher",
            session_id="test"
        )
        
        performance.monitor.record_query(test_metrics)
        
        # Generate report
        report = performance.monitor.get_performance_report()
        print(json.dumps(report, indent=2))
        
        await performance.shutdown()
    
    asyncio.run(main())
