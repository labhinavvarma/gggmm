from mcp.server.fastmcp.prompts.base import Message
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import Prompt
import mcp.types as types

# Add these critical imports for proper message formatting
from mcp.types import TextContent, ImageContent, EmbeddedResource

# The rest of your imports
from functools import partial
import sys
import traceback
import time
