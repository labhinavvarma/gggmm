import re
import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Your existing imports
from dependencies import SnowFlakeConnector
from llm_chat_wrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("custom_react_agent")

@dataclass
class AgentStep:
    """Represents a single step in the React process"""
    step_number: int
    step_type: str  # 'thought', 'action', 'observation', 'final_answer'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    """Complete state of the agent during execution"""
    original_question: str
    steps: List[AgentStep] = field(default_factory=list)
    tools_available: List[str] = field(default_factory=list)
    current_step: int = 0
    max_iterations: int = 10
    is_complete: bool = False
    final_answer: str = ""
    session_id: str = ""
    execution_time: float = 0.0

class CustomReactAgent:
    """Custom React Agent with full process visibility and Snowflake Cortex integration"""
    
    def __init__(self, tools: List = None, max_iterations: int = 10, verbose: bool = True):
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.tool_names = list(self.tools.keys())
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize Snowflake connection
        self.sf_conn = None
        self.model = None
        self._init_snowflake()
        
        # React prompt template
        self.react_prompt = self._create_react_prompt()
        
    def _init_snowflake(self):
        """Initialize Snowflake connection and model"""
        try:
            self._print_process("🔄 INITIALIZING SNOWFLAKE CONNECTION", "system")
            
            self.sf_conn = SnowFlakeConnector.get_conn('aedl', '')
            
            self.model = ChatSnowflakeCortex(
                model="claude-4-sonnet",
                cortex_function="complete",
                session=Session.builder.configs({"connection": self.sf_conn}).getOrCreate()
            )
            
            self._print_process("✅ SNOWFLAKE CONNECTION ESTABLISHED", "system")
            logger.info("Snowflake connection and model initialized successfully")
            
        except Exception as e:
            self._print_process(f"❌ SNOWFLAKE INITIALIZATION FAILED: {e}", "error")
            logger.error(f"Snowflake initialization failed: {e}")
            raise
    
    def _create_react_prompt(self) -> str:
        """Create the React prompt template"""
        if self.tools:
            tools_desc = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
            tools_section = f"""You have access to these tools:

{tools_desc}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{', '.join(self.tool_names)}]
Action Input: the input to the action (as a JSON object)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
        else:
            tools_section = """You are a helpful AI assistant. Since no tools are available, you should answer based on your knowledge.

Use the following format:

Question: the input question you must answer  
Thought: you should think about the question and how to answer it
Final Answer: the final answer to the original input question"""
        
        return f"""You are a helpful AI assistant that follows the ReAct (Reasoning and Acting) framework.

{tools_section}

Begin!

Question: {{question}}"""

    def _print_process(self, message: str, category: str = "info"):
        """Print process messages with formatting"""
        if not self.verbose:
            return
            
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Different formatting for different categories
        if category == "system":
            print(f"\n{'='*80}")
            print(f"🚀 {message}")
            print(f"⏰ {timestamp}")
            print(f"{'='*80}")
        elif category == "error":
            print(f"\n{'❌'*40}")
            print(f"💥 {message}")
            print(f"⏰ {timestamp}")
            print(f"{'❌'*40}")
        elif category == "step":
            print(f"\n{'📍'*20}")
            print(f"📍 {message}")
            print(f"⏰ {timestamp}")
            print(f"{'📍'*20}")
        elif category == "llm":
            print(f"\n{'🤖'*15}")
            print(f"🤖 {message}")
            print(f"⏰ {timestamp}")
            print(f"{'🤖'*15}")
        elif category == "tool":
            print(f"\n{'🔧'*15}")
            print(f"🔧 {message}")
            print(f"⏰ {timestamp}")
            print(f"{'🔧'*15}")
        else:
            print(f"\n💬 {message} [{timestamp}]")

    def _print_step_detail(self, step: AgentStep):
        """Print detailed step information"""
        if not self.verbose:
            return
            
        print(f"\n{'='*60}")
        print(f"📋 STEP {step.step_number} DETAILS - {step.step_type.upper()}")
        print(f"⏰ Timestamp: {step.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"{'='*60}")
        print(f"📝 Content:")
        print(f"{step.content}")
        
        if step.metadata:
            print(f"\n📊 Metadata:")
            for key, value in step.metadata.items():
                if isinstance(value, (dict, list)):
                    print(f"  {key}: {json.dumps(value, indent=4)}")
                else:
                    print(f"  {key}: {value}")
        print(f"{'='*60}\n")

    def _parse_llm_output(self, output: str) -> Tuple[str, str, Dict]:
        """Parse LLM output to extract thought, action, and action input"""
        self._print_process("PARSING LLM OUTPUT", "llm")
        print(f"📥 Raw LLM Output (first 300 chars):")
        print(f"{'─'*50}")
        print(f"{output[:300]}{'...' if len(output) > 300 else ''}")
        print(f"{'─'*50}")
        
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|$)', output, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Check for final answer
        final_answer_match = re.search(r'Final Answer:\s*(.*)', output, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            self._print_process(f"FOUND FINAL ANSWER: {final_answer[:100]}...", "step")
            return "final_answer", final_answer, {"thought": thought}
        
        # Extract action (only if tools are available)
        if self.tools:
            action_match = re.search(r'Action:\s*(.*?)(?=\n|$)', output, re.IGNORECASE)
            action = action_match.group(1).strip() if action_match else ""
            
            # Extract action input
            action_input_match = re.search(r'Action Input:\s*(.*?)(?=\n(?:Observation|Thought|Action|Final Answer):|$)', output, re.DOTALL | re.IGNORECASE)
            action_input_str = action_input_match.group(1).strip() if action_input_match else ""
            
            # Parse action input as JSON
            action_input = {}
            if action_input_str:
                try:
                    action_input = json.loads(action_input_str)
                    self._print_process(f"PARSED ACTION INPUT AS JSON: {action_input}", "info")
                except json.JSONDecodeError:
                    # Try to extract simple key-value or just use as string
                    self._print_process("FAILED TO PARSE AS JSON, USING RAW STRING", "info")
                    action_input = {"input": action_input_str}
            
            print(f"🔍 Parsing Results:")
            print(f"   Thought: {thought[:50]}{'...' if len(thought) > 50 else ''}")
            print(f"   Action: {action}")
            print(f"   Action Input: {action_input}")
            
            if action and action in self.tools:
                self._print_process(f"VALID ACTION FOUND: {action}", "step")
                return "action", action, {"thought": thought, "action_input": action_input}
            else:
                self._print_process(f"INVALID ACTION '{action}' - Available: {self.tool_names}", "error")
                return "thought", thought, {"raw_output": output, "attempted_action": action}
        else:
            # No tools available, just return thought
            return "thought", thought, {"raw_output": output}

    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute a tool and return the result"""
        self._print_process(f"EXECUTING TOOL: {tool_name}", "tool")
        print(f"📥 Tool Input:")
        print(f"{json.dumps(tool_input, indent=2)}")
        
        try:
            tool = self.tools[tool_name]
            
            # Call the tool with appropriate arguments
            if hasattr(tool, 'coroutine'):
                result = await tool.coroutine(**tool_input)
            else:
                result = tool.func(**tool_input)
            
            self._print_process(f"TOOL EXECUTION SUCCESSFUL", "tool")
            print(f"📤 Tool Output (first 500 chars):")
            print(f"{'─'*50}")
            result_str = str(result)
            print(f"{result_str[:500]}{'...' if len(result_str) > 500 else ''}")
            print(f"{'─'*50}")
            
            return result_str
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            self._print_process(f"TOOL EXECUTION FAILED: {error_msg}", "error")
            logger.error(f"Tool {tool_name} failed: {e}")
            return error_msg

    async def _call_llm(self, prompt: str) -> str:
        """Call the Snowflake Cortex LLM model"""
        self._print_process("CALLING SNOWFLAKE CORTEX LLM", "llm")
        print(f"📊 Prompt Stats:")
        print(f"   Length: {len(prompt)} characters")
        print(f"   Lines: {len(prompt.split(chr(10)))}")
        
        print(f"\n📥 Prompt Content (first 400 chars):")
        print(f"{'─'*60}")
        print(f"{prompt[:400]}{'...' if len(prompt) > 400 else ''}")
        print(f"{'─'*60}")
        
        try:
            # Create message for the model
            messages = [HumanMessage(content=prompt)]
            
            # Call the model
            self._print_process("SENDING REQUEST TO SNOWFLAKE...", "llm")
            response = await self.model.ainvoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            self._print_process("LLM RESPONSE RECEIVED", "llm")
            print(f"📊 Response Stats:")
            print(f"   Length: {len(result)} characters")
            print(f"   Lines: {len(result.split(chr(10)))}")
            
            return result
            
        except Exception as e:
            error_msg = f"LLM call failed: {str(e)}"
            self._print_process(f"LLM CALL FAILED: {error_msg}", "error")
            logger.error(f"Snowflake LLM call failed: {e}")
            return f"Error: {error_msg}"

    async def run(self, question: str, session_id: str = "default") -> AgentState:
        """Run the React agent"""
        start_time = datetime.now()
        
        self._print_process("STARTING CUSTOM REACT AGENT EXECUTION", "system")
        print(f"❓ Question: {question}")
        print(f"🆔 Session ID: {session_id}")
        print(f"🛠️  Available Tools: {self.tool_names if self.tool_names else 'None'}")
        print(f"🔄 Max Iterations: {self.max_iterations}")
        
        # Initialize state
        state = AgentState(
            original_question=question,
            tools_available=self.tool_names,
            max_iterations=self.max_iterations,
            session_id=session_id
        )
        
        # Start with the initial prompt
        current_prompt = self.react_prompt.format(question=question)
        
        for iteration in range(self.max_iterations):
            self._print_process(f"ITERATION {iteration + 1}/{self.max_iterations}", "step")
            state.current_step = iteration + 1
            
            # Call LLM
            llm_output = await self._call_llm(current_prompt)
            
            # Parse the output
            step_type, content, metadata = self._parse_llm_output(llm_output)
            
            # Create and log step
            step = AgentStep(
                step_number=state.current_step,
                step_type=step_type,
                content=content,
                metadata=metadata or {}
            )
            
            state.steps.append(step)
            self._print_step_detail(step)
            
            # Handle different step types
            if step_type == "final_answer":
                state.final_answer = content
                state.is_complete = True
                self._print_process(f"FINAL ANSWER REACHED: {content}", "step")
                break
                
            elif step_type == "action" and self.tools:
                # Execute the tool
                tool_name = content
                tool_input = metadata.get("action_input", {})
                
                observation = await self._execute_tool(tool_name, tool_input)
                
                # Create observation step
                obs_step = AgentStep(
                    step_number=state.current_step,
                    step_type="observation",
                    content=observation,
                    metadata={"tool_used": tool_name, "tool_input": tool_input}
                )
                
                state.steps.append(obs_step)
                self._print_step_detail(obs_step)
                
                # Update prompt with the observation
                current_prompt += f"\nThought: {metadata.get('thought', '')}"
                current_prompt += f"\nAction: {tool_name}"
                current_prompt += f"\nAction Input: {json.dumps(tool_input)}"
                current_prompt += f"\nObservation: {observation}"
                
            else:
                # Just a thought, continue
                current_prompt += f"\nThought: {content}"
                
                # If no tools are available, try to get final answer on next iteration
                if not self.tools:
                    current_prompt += "\nPlease provide your final answer."
        
        # Calculate execution time
        end_time = datetime.now()
        state.execution_time = (end_time - start_time).total_seconds()
        
        # Check if we completed successfully
        if not state.is_complete:
            self._print_process(f"REACHED MAX ITERATIONS ({self.max_iterations}) WITHOUT FINAL ANSWER", "error")
            state.final_answer = "I was unable to complete the task within the maximum number of iterations."
        
        # Print final summary
        self._print_final_summary(state)
        
        return state

    def _print_final_summary(self, state: AgentState):
        """Print a summary of the agent execution"""
        self._print_process("EXECUTION COMPLETE - GENERATING SUMMARY", "system")
        
        print(f"\n{'🎯 EXECUTION SUMMARY'}")
        print("=" * 80)
        print(f"❓ Original Question: {state.original_question}")
        print(f"⏱️  Total Execution Time: {state.execution_time:.2f} seconds")
        print(f"📊 Total Steps: {len(state.steps)}")
        print(f"🔄 Iterations Used: {state.current_step}/{state.max_iterations}")
        print(f"✅ Completed Successfully: {'Yes' if state.is_complete else 'No'}")
        print(f"🎯 Final Answer: {state.final_answer}")
        
        # Step breakdown
        step_types = {}
        for step in state.steps:
            step_types[step.step_type] = step_types.get(step.step_type, 0) + 1
        
        print(f"📋 Step Types Breakdown: {step_types}")
        print(f"🛠️  Tools Available: {len(state.tools_available)} ({', '.join(state.tools_available) if state.tools_available else 'None'})")
        print("=" * 80)

# Example usage - Clean React agent without any tools
async def example_no_tools():
    """Example of using the agent without any tools"""
    print("🧪 TESTING CUSTOM REACT AGENT WITHOUT TOOLS")
    
    # Create agent without tools
    agent = CustomReactAgent(tools=None, max_iterations=5, verbose=True)
    
    # Run with a simple question
    question = "What are the key principles of database design?"
    result = await agent.run(question, session_id="no_tools_test")
    
    print(f"\n🏁 FINAL RESULT:")
    print(f"Answer: {result.final_answer}")
    return result

# Example usage - Adding your own tools
async def example_with_custom_tool():
    """Example of using the agent with a custom tool"""
    print("🧪 TESTING CUSTOM REACT AGENT WITH CUSTOM TOOL")
    
    # Define a simple custom tool
    @tool("calculator")
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression"""
        try:
            # Simple and safe evaluation for basic math
            result = eval(expression.replace("^", "**"))
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    # Create agent with custom tool
    agent = CustomReactAgent(tools=[calculator], max_iterations=5, verbose=True)
    
    # Run with a math question
    question = "What is 25 * 48 + 120 / 6?"
    result = await agent.run(question, session_id="custom_tool_test")
    
    print(f"\n🏁 FINAL RESULT:")
    print(f"Answer: {result.final_answer}")
    return result

if __name__ == "__main__":
    print("🚀 CUSTOM REACT AGENT EXAMPLES")
    print("=" * 50)
    
    # Run example without tools
    result1 = asyncio.run(example_no_tools())
    
    print("\n" + "=" * 50)
    
    # Run example with custom tool
    result2 = asyncio.run(example_with_custom_tool())
