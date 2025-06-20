#!/bin/bash
# Run the MCP Analyzer server locally
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
