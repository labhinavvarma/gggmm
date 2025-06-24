
# langgraph_agent.py

import requests
import json
import uuid
from datetime import datetime, date, timedelta
import random
import logging
from typing import TypedDict, Dict, Any

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# --- Agent Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - AGENT - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM Configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0" # Placeholder
MODEL = "llama3.1-70b"
SYS_MSG = "You are a healthcare AI assistant. Your only job is to answer the user's question based *strictly* on the JSON data provided in the 'CONTEXT' section. If the answer is not in the context, say so directly. Do not use outside knowledge."

# --- State Definition for the Graph ---
class AgentState(TypedDict):
    """Defines the state that is passed between nodes in the LangGraph."""
    raw_patient_record: Dict[str, Any]
    user_question: str
    processed_knowledge_base: Dict[str, Any]
    llm_response: str
    error: str | None

class LangGraphRAGAgent:
    """
    A stable, standalone agent that uses LangGraph to manage an internal
    data processing and question-answering workflow.
    """
    def __init__(self):
        """Initializes the agent and compiles the LangGraph workflow."""
        self.graph = self._build_graph()
        logger.info("LangGraphRAGAgent initialized with a compiled workflow.")

    def _build_graph(self):
        """Defines and compiles the agent's internal processing graph."""
        workflow = StateGraph(AgentState)

        # Define the nodes (steps in the pipeline)
        workflow.add_node("start_processing", self._start_processing)
        workflow.add_node("build_knowledge_base", self._build_knowledge_base)
        workflow.add_node("call_llm_with_context", self._call_llm_with_context)

        # Define the edges (the flow of the pipeline)
        workflow.add_edge(START, "start_processing")
        workflow.add_edge("start_processing", "build_knowledge_base")
        workflow.add_edge("build_knowledge_base", "call_llm_with_context")
        workflow.add_edge("call_llm_with_context", END)

        # Compile the graph into a runnable object
        return workflow.compile()

    def invoke(self, patient_record: dict, question: str) -> str:
        """
        The single public method to run the entire agent workflow.
        """
        if not isinstance(patient_record, dict) or not patient_record:
            return "Error: Invalid or empty patient record provided."
        if not isinstance(question, str) or not question:
            return "Error: A question must be provided."

        logger.info("Invoking agent workflow...")
        initial_state = {
            "raw_patient_record": patient_record,
            "user_question": question,
            "processed_knowledge_base": None,
            "llm_response": "",
            "error": None
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        if final_state.get("error"):
            logger.error(f"Workflow ended with an error: {final_state['error']}")
            return final_state["error"]
        
        logger.info("Workflow completed successfully.")
        return final_state["llm_response"]

    # --- Node Implementations (Private Methods) ---

    def _start_processing(self, state: AgentState) -> AgentState:
        """Node 1: Logs the start of the process."""
        logger.info("Node [start_processing]: Beginning workflow.")
        # This node could be expanded to perform initial validation if needed
        return state

    def _build_knowledge_base(self, state: AgentState) -> AgentState:
        """Node 2: Takes the raw record and enriches it into a knowledge base."""
        logger.info("Node [build_knowledge_base]: Enriching raw data.")
        try:
            raw_record = state["raw_patient_record"]
            
            # Call helper functions to generate enriched data
            health_entities = self._extract_health_entities(raw_record)
            medical_claims = self._generate_claims("Medical", 5)
            pharmacy_claims = self._generate_claims("Pharmacy", 3)
            recommendations = self._generate_recommendations(health_entities)
            
            # Assemble the final knowledge base
            knowledge_base = {
                "patient_summary": raw_record,
                "inferred_health_factors": health_entities,
                "medical_claims_history": medical_claims,
                "pharmacy_claims_history": pharmacy_claims,
                "proactive_recommendations": recommendations,
                "report_generated_at": datetime.now().isoformat()
            }
            return {**state, "processed_knowledge_base": knowledge_base}
        except Exception as e:
            logger.error(f"Error in _build_knowledge_base: {e}")
            return {**state, "error": f"Failed to build knowledge base: {e}"}

    def _call_llm_with_context(self, state: AgentState) -> AgentState:
        """Node 3: Constructs the final prompt and queries the LLM."""
        logger.info("Node [call_llm_with_context]: Calling LLM.")
        try:
            user_query = state["user_question"]
            json_context = state["processed_knowledge_base"]
            
            json_blob = f"CONTEXT:\n{json.dumps(json_context, indent=2)}"
            full_prompt = f"{SYS_MSG}\n\n{json_blob}\n\nUser Question: {user_query}"
            
            payload = {"query": {"method": "cortex", "model": MODEL, "prompt": {"messages": [{"role": "user", "content": full_prompt}]}}}
            headers = {"Content-Type": "application/json", "Authorization": f'Snowflake Token="{API_KEY}"'}
            
            response = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=45)
            response.raise_for_status()
            
            raw_response = response.text
            answer = raw_response.partition("end_of_stream")[0].strip() if "end_of_stream" in raw_response else raw_response.strip()
            
            return {**state, "llm_response": answer}
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API call failed: {e}")
            return {**state, "error": f"AI model communication error: {e}"}

    # --- Data Processing Helpers ---

    def _extract_health_entities(self, data: dict) -> dict:
        """Infers health entities from basic patient data."""
        entities = {"age": "N/A", "age_group": "N/A", "inferred_risks": []}
        if data.get("date_of_birth"):
            try:
                age = (date.today() - datetime.strptime(data["date_of_birth"], '%Y-%m-%d').date()).days // 365
                entities["age"] = age
                if age >= 65:
                    entities["age_group"] = "Senior"
                    entities["inferred_risks"].append("Higher risk for age-related conditions.")
                elif 40 <= age < 65:
                    entities["age_group"] = "Adult"
            except (ValueError, TypeError):
                pass
        return entities

    def _generate_claims(self, claim_type: str, count: int) -> list:
        """Generates a list of mock claims."""
        return [{"description": f"{claim_type} Service #{i + 1}", "date": (date.today() - timedelta(days=random.randint(20, 1200))).isoformat()} for i in range(count)]

    def _generate_recommendations(self, entities: dict) -> dict:
        """Generates proactive recommendations."""
        recs = {"preventive_care": []}
        if entities["age_group"] == "Senior":
            recs["preventive_care"].extend(["Annual wellness visit", "Bone density screening"])
        if entities["age"] != "N/A" and entities["age"] > 50:
            recs["preventive_care"].append("Blood pressure and cholesterol screenings.")
        return recs
