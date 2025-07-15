# langgraph_mcpserver.py - Specialized MCP server for LangGraph Neo4j Agent

import json
import re
import logging
import asyncio
import sys
import time
from typing import Any, Optional, Dict, List
from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncTransaction
from pydantic import Field

# Enhanced logging for LangGraph integration
logger = logging.getLogger("langgraph_neo4j_mcp")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class QueryMetrics:
    """Track query performance metrics."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.syntax_fixes_applied = 0
        self.avg_execution_time = 0.0
        self.query_history = []
    
    def record_query(self, success: bool, execution_time: float, had_syntax_fix: bool = False):
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        if had_syntax_fix:
            self.syntax_fixes_applied += 1
        
        # Update average execution time
        self.avg_execution_time = (
            (self.avg_execution_time * (self.total_queries - 1) + execution_time) / self.total_queries
        )
    
    def get_stats(self) -> Dict:
        success_rate = (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": round(success_rate, 2),
            "syntax_fixes_applied": self.syntax_fixes_applied,
            "avg_execution_time_ms": round(self.avg_execution_time * 1000, 2)
        }

# Global metrics instance
query_metrics = QueryMetrics()

def _is_write_query(query: str) -> bool:
    """Check if query is a write operation."""
    return bool(re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH)\b", query, re.IGNORECASE))

def _apply_syntax_fixes(query: str) -> tuple[str, bool]:
    """Apply modern Neo4j syntax fixes and return (fixed_query, was_modified)."""
    original_query = query
    fixed_query = query
    
    # Common syntax fixes for Neo4j 5.x
    fixes = [
        # Size function fixes
        (r"size\(\s*\(([^)]+)\)\s*-\s*\[\s*\]\s*-\s*\(\s*\)\s*\)", r"COUNT { (\1)-[]-() }"),
        (r"size\(\s*\(([^)]+)\)\s*-\s*\[([^\]]*)\]\s*-\s*\(([^)]*)\)\s*\)", r"COUNT { (\1)-[\2]-(\3) }"),
        
        # Length function fixes
        (r"length\(\s*\(([^)]+)\)\s*-\s*\[\s*\*\s*\]\s*-\s*\(([^)]*)\)\s*\)", r"COUNT { (\1)-[*]-(\2) }"),
        
        # Has function fixes
        (r"has\(\s*([^)]+)\s*\)", r"\1 IS NOT NULL"),
        
        # Legacy relationship patterns
        (r"-->\s*\(\s*\)", r"-()"),
        (r"<--\s*\(\s*\)", r"<-()-"),
        
        # Add LIMIT to potentially expensive queries
        (r"(ORDER BY .+?)(\s*$)", r"\1 LIMIT 100\2"),
    ]
    
    for pattern, replacement in fixes:
        new_query = re.sub(pattern, replacement, fixed_query, flags=re.IGNORECASE)
        if new_query != fixed_query:
            logger.info(f"Applied syntax fix: {pattern}")
            fixed_query = new_query
    
    was_modified = fixed_query != original_query
    return fixed_query, was_modified

async def _execute_read_with_metrics(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> tuple[str, float, bool]:
    """Execute read query with performance metrics."""
    start_time = time.time()
    
    try:
        # Apply syntax fixes
        fixed_query, had_syntax_fix = _apply_syntax_fixes(query)
        
        result = await tx.run(fixed_query, params or {})
        
        # Collect all records using async iteration
        records = []
        async for record in result:
            records.append(record.data())
        
        execution_time = time.time() - start_time
        query_metrics.record_query(True, execution_time, had_syntax_fix)
        
        return json.dumps(records, default=str), execution_time, had_syntax_fix
        
    except Exception as e:
        execution_time = time.time() - start_time
        query_metrics.record_query(False, execution_time)
        logger.error(f"Read query failed: {e}")
        raise

async def _execute_write_with_metrics(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> tuple[str, float, bool]:
    """Execute write query with performance metrics."""
    start_time = time.time()
    
    try:
        # Apply syntax fixes
        fixed_query, had_syntax_fix = _apply_syntax_fixes(query)
        
        result = await tx.run(fixed_query, params or {})
        summary = await result.consume()
        
        execution_time = time.time() - start_time
        query_metrics.record_query(True, execution_time, had_syntax_fix)
        
        return json.dumps(summary.counters._raw_data, default=str), execution_time, had_syntax_fix
        
    except Exception as e:
        execution_time = time.time() - start_time
        query_metrics.record_query(False, execution_time)
        logger.error(f"Write query failed: {e}")
        raise

def create_langgraph_mcp_server(driver: AsyncDriver, database: str) -> FastMCP:
    """Create enhanced MCP server optimized for LangGraph agent."""
    mcp = FastMCP("langgraph-neo4j-cypher")

    # =================== BASIC TOOLS ===================

    @mcp.tool(name="health_check")
    async def health_check() -> str:
        """Enhanced health check with detailed information."""
        try:
            async with driver.session(database=database) as session:
                start_time = time.time()
                result = await session.run("RETURN 1 as health, datetime() as timestamp")
                record = await result.single()
                response_time = (time.time() - start_time) * 1000
                
                health_info = {
                    "status": "healthy",
                    "database": database,
                    "timestamp": str(record["timestamp"]),
                    "response_time_ms": round(response_time, 2),
                    "message": "LangGraph MCP server connection successful",
                    "server_type": "langgraph-optimized"
                }
                
                return json.dumps(health_info)
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="get_metrics")
    async def get_metrics() -> str:
        """Get server performance metrics."""
        try:
            metrics = query_metrics.get_stats()
            metrics["server_type"] = "langgraph-optimized"
            metrics["database"] = database
            return json.dumps(metrics)
        except Exception as e:
            error_msg = f"Failed to get metrics: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="reset_metrics")
    async def reset_metrics() -> str:
        """Reset performance metrics."""
        try:
            query_metrics.reset()
            return json.dumps({
                "status": "success",
                "message": "Metrics reset successfully",
                "timestamp": time.time()
            })
        except Exception as e:
            error_msg = f"Failed to reset metrics: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    # =================== CORE QUERY TOOLS ===================

    @mcp.tool(name="execute_read_query")
    async def execute_read_query(
        query: str = Field(..., description="Cypher read query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters"),
        apply_fixes: bool = Field(True, description="Whether to apply automatic syntax fixes")
    ) -> str:
        """Enhanced read query execution with automatic fixes."""
        if _is_write_query(query):
            error_msg = "Write operations not allowed in read tool"
            logger.warning(f"Rejected write query: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                if apply_fixes:
                    result_text, exec_time, had_fix = await session.execute_read(
                        _execute_read_with_metrics, query, params
                    )
                    
                    # Add metadata about execution
                    try:
                        parsed_result = json.loads(result_text)
                        enhanced_result = {
                            "data": parsed_result,
                            "metadata": {
                                "execution_time_ms": round(exec_time * 1000, 2),
                                "syntax_fixes_applied": had_fix,
                                "record_count": len(parsed_result) if isinstance(parsed_result, list) else 1,
                                "query_type": "read"
                            }
                        }
                        return json.dumps(enhanced_result, default=str)
                    except json.JSONDecodeError:
                        return result_text
                else:
                    # Execute without fixes (for testing original queries)
                    result = await session.run(query, params or {})
                    records = []
                    async for record in result:
                        records.append(record.data())
                    return json.dumps(records, default=str)
                    
        except Exception as e:
            error_msg = f"Read query execution failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="execute_write_query")
    async def execute_write_query(
        query: str = Field(..., description="Cypher write query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters"),
        apply_fixes: bool = Field(True, description="Whether to apply automatic syntax fixes")
    ) -> str:
        """Enhanced write query execution with automatic fixes."""
        if not _is_write_query(query):
            error_msg = "Only write operations allowed in write tool"
            logger.warning(f"Rejected read query in write tool: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                if apply_fixes:
                    result_text, exec_time, had_fix = await session.execute_write(
                        _execute_write_with_metrics, query, params
                    )
                    
                    # Add metadata about execution
                    try:
                        parsed_result = json.loads(result_text)
                        enhanced_result = {
                            "counters": parsed_result,
                            "metadata": {
                                "execution_time_ms": round(exec_time * 1000, 2),
                                "syntax_fixes_applied": had_fix,
                                "query_type": "write"
                            }
                        }
                        return json.dumps(enhanced_result, default=str)
                    except json.JSONDecodeError:
                        return result_text
                else:
                    # Execute without fixes
                    result = await session.run(query, params or {})
                    summary = await result.consume()
                    return json.dumps(summary.counters._raw_data, default=str)
                    
        except Exception as e:
            error_msg = f"Write query execution failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    # =================== SCHEMA ANALYSIS TOOLS ===================

    @mcp.tool(name="analyze_schema")
    async def analyze_schema() -> str:
        """Comprehensive schema analysis for LangGraph context."""
        try:
            async with driver.session(database=database) as session:
                schema_analysis = {}
                
                # Get labels with counts
                labels_result = await session.run("""
                    CALL db.labels() YIELD label
                    CALL {
                        WITH label
                        CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
                        YIELD value
                        RETURN value.count as count
                    }
                    RETURN label, count
                    ORDER BY count DESC
                """)
                
                labels_with_counts = []
                async for record in labels_result:
                    labels_with_counts.append({
                        "label": record["label"],
                        "count": record["count"]
                    })
                
                # Get relationship types with counts
                rel_result = await session.run("""
                    CALL db.relationshipTypes() YIELD relationshipType
                    CALL {
                        WITH relationshipType
                        CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {})
                        YIELD value
                        RETURN value.count as count
                    }
                    RETURN relationshipType, count
                    ORDER BY count DESC
                """)
                
                relationships_with_counts = []
                async for record in rel_result:
                    relationships_with_counts.append({
                        "type": record["relationshipType"],
                        "count": record["count"]
                    })
                
                # Get property analysis
                prop_result = await session.run("""
                    CALL db.labels() YIELD label
                    CALL {
                        WITH label
                        CALL apoc.meta.nodeTypeProperties()
                        YIELD nodeType, propertyName, propertyTypes
                        WHERE nodeType = ':' + label
                        RETURN propertyName, propertyTypes
                    }
                    RETURN label, collect({property: propertyName, types: propertyTypes}) as properties
                """)
                
                node_properties = {}
                async for record in prop_result:
                    node_properties[record["label"]] = record["properties"]
                
                schema_analysis = {
                    "labels": labels_with_counts,
                    "relationships": relationships_with_counts,
                    "node_properties": node_properties,
                    "total_labels": len(labels_with_counts),
                    "total_relationship_types": len(relationships_with_counts),
                    "analysis_timestamp": time.time()
                }
                
                return json.dumps(schema_analysis, default=str)
                
        except Exception as e:
            # Fallback to basic schema analysis if APOC is not available
            logger.warning(f"Advanced schema analysis failed, using basic approach: {e}")
            return await basic_schema_analysis(session)

    async def basic_schema_analysis(self, session) -> str:
        """Basic schema analysis without APOC procedures."""
        try:
            # Basic labels
            labels_result = await session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
            labels_record = await labels_result.single()
            labels = labels_record["labels"] if labels_record else []
            
            # Basic relationships
            rel_result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
            rel_record = await rel_result.single()
            relationships = rel_record["types"] if rel_record else []
            
            # Node and relationship counts
            node_count_result = await session.run("MATCH (n) RETURN count(n) as nodeCount")
            node_count_record = await node_count_result.single()
            node_count = node_count_record["nodeCount"] if node_count_record else 0
            
            rel_count_result = await session.run("MATCH ()-[r]->() RETURN count(r) as relCount")
            rel_count_record = await rel_count_result.single()
            rel_count = rel_count_record["relCount"] if rel_count_record else 0
            
            basic_analysis = {
                "labels": [{"label": label, "count": "unknown"} for label in labels],
                "relationships": [{"type": rel_type, "count": "unknown"} for rel_type in relationships],
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "total_labels": len(labels),
                "total_relationship_types": len(relationships),
                "analysis_type": "basic",
                "analysis_timestamp": time.time()
            }
            
            return json.dumps(basic_analysis, default=str)
            
        except Exception as e:
            error_msg = f"Schema analysis failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="get_sample_data")
    async def get_sample_data(
        label: Optional[str] = Field(None, description="Specific node label to sample"),
        limit: int = Field(5, description="Number of samples to return")
    ) -> str:
        """Get sample data for schema understanding."""
        try:
            async with driver.session(database=database) as session:
                if label:
                    # Sample specific label
                    query = f"MATCH (n:{label}) RETURN n LIMIT $limit"
                    result = await session.run(query, {"limit": limit})
                else:
                    # Sample all nodes
                    query = "MATCH (n) RETURN n LIMIT $limit"
                    result = await session.run(query, {"limit": limit})
                
                samples = []
                async for record in result:
                    samples.append(record.data())
                
                sample_data = {
                    "samples": samples,
                    "count": len(samples),
                    "label_filter": label,
                    "limit": limit
                }
                
                return json.dumps(sample_data, default=str)
                
        except Exception as e:
            error_msg = f"Sample data retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    # =================== QUERY VALIDATION TOOLS ===================

    @mcp.tool(name="validate_query")
    async def validate_query(
        query: str = Field(..., description="Cypher query to validate"),
        apply_fixes: bool = Field(True, description="Whether to suggest fixes")
    ) -> str:
        """Validate Cypher query syntax and suggest improvements."""
        try:
            validation_result = {
                "original_query": query,
                "is_valid": True,
                "issues": [],
                "suggestions": [],
                "estimated_performance": "unknown"
            }
            
            # Check for common issues
            issues = []
            suggestions = []
            
            # Check for deprecated syntax
            if re.search(r"size\(\s*\([^)]+\)\s*-\s*\[\s*\]\s*-\s*\(\s*\)\s*\)", query, re.IGNORECASE):
                issues.append("Uses deprecated size() function")
                suggestions.append("Replace size((n)-[]->()) with COUNT { (n)-[]-() }")
            
            if re.search(r"has\(\s*\w+\s*\)", query, re.IGNORECASE):
                issues.append("Uses deprecated has() function")
                suggestions.append("Replace has(property) with property IS NOT NULL")
            
            # Check for performance issues
            if "ORDER BY" in query.upper() and "LIMIT" not in query.upper():
                issues.append("ORDER BY without LIMIT may cause performance issues")
                suggestions.append("Consider adding LIMIT clause")
            
            if query.count("MATCH") > 3:
                issues.append("Multiple MATCH clauses may impact performance")
                suggestions.append("Consider using WITH clauses or subqueries")
            
            # Estimate query complexity
            complexity_score = 0
            if "ORDER BY" in query.upper():
                complexity_score += 2
            if re.search(r"OPTIONAL\s+MATCH", query, re.IGNORECASE):
                complexity_score += 1
            complexity_score += query.count("MATCH")
            
            if complexity_score <= 2:
                performance_estimate = "low"
            elif complexity_score <= 5:
                performance_estimate = "medium"
            else:
                performance_estimate = "high"
            
            validation_result.update({
                "issues": issues,
                "suggestions": suggestions,
                "estimated_performance": performance_estimate,
                "complexity_score": complexity_score,
                "is_valid": len(issues) == 0
            })
            
            # Apply fixes if requested
            if apply_fixes and issues:
                fixed_query, _ = _apply_syntax_fixes(query)
                validation_result["suggested_query"] = fixed_query
            
            return json.dumps(validation_result, default=str)
            
        except Exception as e:
            error_msg = f"Query validation failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="explain_query")
    async def explain_query(
        query: str = Field(..., description="Cypher query to explain")
    ) -> str:
        """Get query execution plan and analysis."""
        try:
            async with driver.session(database=database) as session:
                # Get query plan
                explain_query = f"EXPLAIN {query}"
                result = await session.run(explain_query)
                
                plan_info = []
                async for record in result:
                    plan_info.append(record.data())
                
                explanation = {
                    "query": query,
                    "execution_plan": plan_info,
                    "analysis_timestamp": time.time()
                }
                
                return json.dumps(explanation, default=str)
                
        except Exception as e:
            error_msg = f"Query explanation failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    # =================== UTILITY TOOLS ===================

    @mcp.tool(name="count_by_label")
    async def count_by_label() -> str:
        """Count nodes by label."""
        try:
            async with driver.session(database=database) as session:
                result = await session.run("""
                    CALL db.labels() YIELD label
                    CALL {
                        WITH label
                        MATCH (n)
                        WHERE label IN labels(n)
                        RETURN count(n) as count
                    }
                    RETURN label, count
                    ORDER BY count DESC
                """)
                
                counts = []
                async for record in result:
                    counts.append({
                        "label": record["label"],
                        "count": record["count"]
                    })
                
                return json.dumps({
                    "label_counts": counts,
                    "timestamp": time.time()
                }, default=str)
                
        except Exception as e:
            # Fallback method
            async with driver.session(database=database) as session:
                labels_result = await session.run("CALL db.labels() YIELD label RETURN label")
                labels = []
                async for record in labels_result:
                    labels.append(record["label"])
                
                counts = []
                for label in labels:
                    count_result = await session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    count_record = await count_result.single()
                    counts.append({
                        "label": label,
                        "count": count_record["count"] if count_record else 0
                    })
                
                return json.dumps({
                    "label_counts": counts,
                    "method": "fallback",
                    "timestamp": time.time()
                }, default=str)

    @mcp.tool(name="database_summary")
    async def database_summary() -> str:
        """Enhanced database summary with performance metrics."""
        try:
            async with driver.session(database=database) as session:
                # Basic counts
                node_result = await session.run("MATCH (n) RETURN count(n) as nodeCount")
                node_record = await node_result.single()
                node_count = node_record["nodeCount"] if node_record else 0
                
                rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as relCount")
                rel_record = await rel_result.single()
                rel_count = rel_record["relCount"] if rel_record else 0
                
                # Schema info
                labels_result = await session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                labels_record = await labels_result.single()
                labels = labels_record["labels"] if labels_record else []
                
                rel_types_result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
                rel_types_record = await rel_types_result.single()
                rel_types = rel_types_record["types"] if rel_types_record else []
                
                # Performance metrics
                metrics = query_metrics.get_stats()
                
                summary = {
                    "database": database,
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "label_count": len(labels),
                    "labels": labels,
                    "relationship_type_count": len(rel_types),
                    "relationship_types": rel_types,
                    "performance_metrics": metrics,
                    "server_type": "langgraph-optimized",
                    "summary_timestamp": time.time()
                }
                
                return json.dumps(summary, default=str)
                
        except Exception as e:
            error_msg = f"Database summary failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    return mcp

async def run_langgraph_mcp_server():
    """Main server function for LangGraph-optimized MCP server."""
    driver = None
    try:
        logger.info("🚀 Starting LangGraph-optimized MCP server...")
        
        # Create Neo4j driver
        driver = AsyncGraphDatabase.driver(
            "neo4j://10.189.116.237:7687",
            auth=("neo4j", "Vkg5d$F!pLq2@9vRwE="),
        )
        
        # Test connection
        async with driver.session(database="connectiq") as session:
            result = await session.run("RETURN 1 as test, datetime() as timestamp")
            record = await result.single()
            logger.info(f"✅ Neo4j connection established at {record['timestamp']}")
        
        # Create enhanced MCP server
        mcp = create_langgraph_mcp_server(driver, "connectiq")
        logger.info("✅ LangGraph-optimized MCP server created with enhanced tools")
        
        # Initialize metrics
        query_metrics.reset()
        logger.info("📊 Performance metrics initialized")
        
        # Use STDIO transport
        logger.info("🚀 Starting MCP server with STDIO transport")
        await mcp.run_async(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
    finally:
        if driver:
            await driver.close()
            logger.info("Neo4j driver closed")

def main():
    """Main entry point."""
    try:
        asyncio.run(run_langgraph_mcp_server())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
