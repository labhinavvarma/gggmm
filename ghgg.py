from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from loguru import logger
from typing import Any, Dict, List
import numpy as np

# Initialize FastAPI and FastMCP
app = FastAPI(title="MCP JSON Analyzer Server")
mcp = FastMCP("Universal JSON MCP Server", app=app)

def extract_numeric_values(data: Any) -> Dict[str, float]:
    """Extract all numeric values from nested JSON structure"""
    result = {}

    def process_value(value: Any, path: str = ""):
        if isinstance(value, dict):
            for k, v in value.items():
                new_path = f"{path}.{k}" if path else k
                process_value(v, new_path)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                new_path = f"{path}[{i}]"
                process_value(v, new_path)
        else:
            try:
                if isinstance(value, str):
                    value = value.replace(',', '')
                num_value = float(value)
                result[path] = num_value
            except (ValueError, TypeError):
                pass

    process_value(data)
    return result

@mcp.tool()
def analyze_data(data: List[Dict], column: str, operation: str) -> Dict[str, Any]:
    """Analyze numeric column from nested JSON data."""
    try:
        numeric_data = []
        for item in data:
            numeric_values = extract_numeric_values(item)
            if column in numeric_values:
                numeric_data.append(numeric_values[column])

        if not numeric_data:
            raise ValueError(f"No numeric values found for column '{column}'")

        arr = np.array(numeric_data)

        if operation == "sum":
            result = float(np.sum(arr))
        elif operation in ("mean", "average"):
            result = float(np.mean(arr))
        elif operation == "median":
            result = float(np.median(arr))
        elif operation == "min":
            result = float(np.min(arr))
        elif operation == "max":
            result = float(np.max(arr))
        elif operation == "count":
            result = float(len(arr))
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return {"status": "success", "value": result}
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/tool/analyze-data")
async def analyze_data_route(request: Request):
    """Manual HTTP route wrapper for MCP tool"""
    try:
        body = await request.json()
        data = body.get("data")
        column = body.get("column")
        operation = body.get("operation")

        if not data or not column or not operation:
            return JSONResponse(status_code=400, content={"status": "error", "error": "Missing required parameters: data, column, operation"})

        result = analyze_data(data=data, column=column, operation=operation)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})
