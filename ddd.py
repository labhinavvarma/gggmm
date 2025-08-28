# Add to your imports
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add episodic memory storage to HealthAnalysisState
class HealthAnalysisState(TypedDict):
    # ... existing fields ...
    
    # New episodic memory fields
    memidnum: str
    episodic_memory: Dict[str, Any]
    health_entity_comparison: Dict[str, Any]
    temporal_analysis: str

@dataclass
class EpisodicMemoryManager:
    """Manages episodic memory for health entities using memidnum"""
    
    def __init__(self, storage_path: str = "episodic_memory.json"):
        self.storage_path = storage_path
        self.memory_store = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load existing episodic memory from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning(f"Failed to load episodic memory: {e}")
            return {}
    
    def _save_memory(self):
        """Save episodic memory to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.memory_store, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save episodic memory: {e}")
    
    def extract_memidnum(self, mcid_data: Dict[str, Any]) -> Optional[str]:
        """Extract memidnum from mcid data"""
        try:
            # Try different possible locations for memidnum
            possible_paths = [
                ['memidnum'],
                ['member_id'],
                ['mcid'],
                ['patient_id'],
                # Add more paths based on your mcid JSON structure
            ]
            
            for path in possible_paths:
                value = mcid_data
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        break
                else:
                    if value and str(value).strip():
                        return str(value).strip()
            
            # If not found in standard locations, search recursively
            return self._recursive_search_memidnum(mcid_data)
            
        except Exception as e:
            logger.error(f"Error extracting memidnum: {e}")
            return None
    
    def _recursive_search_memidnum(self, data: Any, depth: int = 0) -> Optional[str]:
        """Recursively search for memidnum in nested data"""
        if depth > 5:  # Prevent infinite recursion
            return None
            
        if isinstance(data, dict):
            # Check for keys that might contain member ID
            for key, value in data.items():
                if any(term in key.lower() for term in ['memid', 'member', 'patient', 'mcid']):
                    if isinstance(value, (str, int)) and str(value).strip():
                        return str(value).strip()
                
                # Recurse into nested structures
                result = self._recursive_search_memidnum(value, depth + 1)
                if result:
                    return result
        
        elif isinstance(data, list) and data:
            # Check first few items in list
            for item in data[:3]:
                result = self._recursive_search_memidnum(item, depth + 1)
                if result:
                    return result
        
        return None
    
    def store_health_entities(self, memidnum: str, entities: Dict[str, Any]) -> bool:
        """Store health entities for a given memidnum"""
        try:
            if not memidnum:
                return False
            
            timestamp = datetime.now().isoformat()
            
            if memidnum not in self.memory_store:
                self.memory_store[memidnum] = {
                    "episodes": [],
                    "first_seen": timestamp,
                    "last_updated": timestamp
                }
            
            # Create episode entry
            episode = {
                "timestamp": timestamp,
                "entities": entities,
                "analysis_date": timestamp[:10]  # YYYY-MM-DD
            }
            
            self.memory_store[memidnum]["episodes"].append(episode)
            self.memory_store[memidnum]["last_updated"] = timestamp
            
            # Keep only last 10 episodes to prevent storage bloat
            if len(self.memory_store[memidnum]["episodes"]) > 10:
                self.memory_store[memidnum]["episodes"] = self.memory_store[memidnum]["episodes"][-10:]
            
            self._save_memory()
            logger.info(f"Stored health entities for memidnum: {memidnum}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing health entities: {e}")
            return False
    
    def get_previous_entities(self, memidnum: str) -> Optional[Dict[str, Any]]:
        """Get the most recent previous health entities for comparison"""
        try:
            if not memidnum or memidnum not in self.memory_store:
                return None
            
            episodes = self.memory_store[memidnum]["episodes"]
            if len(episodes) < 2:  # Need at least 2 episodes for comparison
                return None
            
            # Return the second-to-last episode (most recent previous)
            return episodes[-2]
            
        except Exception as e:
            logger.error(f"Error getting previous entities: {e}")
            return None
    
    def compare_health_entities(self, current_entities: Dict[str, Any], 
                              previous_episode: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current and previous health entities"""
        try:
            previous_entities = previous_episode.get("entities", {})
            previous_date = previous_episode.get("analysis_date", "Unknown")
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            comparison = {
                "comparison_metadata": {
                    "previous_date": previous_date,
                    "current_date": current_date,
                    "time_difference_days": self._calculate_days_difference(previous_date, current_date)
                },
                "changes": {
                    "new_conditions": [],
                    "resolved_conditions": [],
                    "changed_values": [],
                    "new_medications": [],
                    "discontinued_medications": [],
                    "stable_conditions": []
                },
                "summary": ""
            }
            
            # Compare medical conditions
            prev_conditions = set(previous_entities.get("medical_conditions", []))
            curr_conditions = set(current_entities.get("medical_conditions", []))
            
            comparison["changes"]["new_conditions"] = list(curr_conditions - prev_conditions)
            comparison["changes"]["resolved_conditions"] = list(prev_conditions - curr_conditions)
            comparison["changes"]["stable_conditions"] = list(prev_conditions & curr_conditions)
            
            # Compare medications
            prev_meds = set(previous_entities.get("medications_identified", []))
            curr_meds = set(current_entities.get("medications_identified", []))
            
            comparison["changes"]["new_medications"] = list(curr_meds - prev_meds)
            comparison["changes"]["discontinued_medications"] = list(prev_meds - curr_meds)
            
            # Compare key health indicators
            key_indicators = ["diabetics", "smoking", "blood_pressure", "age_group"]
            for indicator in key_indicators:
                prev_value = previous_entities.get(indicator)
                curr_value = current_entities.get(indicator)
                
                if prev_value != curr_value and prev_value is not None and curr_value is not None:
                    comparison["changes"]["changed_values"].append({
                        "indicator": indicator,
                        "previous": prev_value,
                        "current": curr_value
                    })
            
            # Generate summary
            comparison["summary"] = self._generate_comparison_summary(comparison)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing health entities: {e}")
            return {"error": f"Comparison failed: {str(e)}"}
    
    def _calculate_days_difference(self, date1: str, date2: str) -> int:
        """Calculate difference in days between two dates"""
        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            return abs((d2 - d1).days)
        except:
            return 0
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> str:
        """Generate a human-readable summary of changes"""
        changes = comparison["changes"]
        metadata = comparison["comparison_metadata"]
        
        summary_parts = []
        
        time_diff = metadata.get("time_difference_days", 0)
        summary_parts.append(f"Health status comparison over {time_diff} days:")
        
        if changes["new_conditions"]:
            summary_parts.append(f"New conditions identified: {', '.join(changes['new_conditions'])}")
        
        if changes["resolved_conditions"]:
            summary_parts.append(f"Previously noted conditions not found: {', '.join(changes['resolved_conditions'])}")
        
        if changes["new_medications"]:
            summary_parts.append(f"New medications: {', '.join(changes['new_medications'])}")
        
        if changes["discontinued_medications"]:
            summary_parts.append(f"Discontinued medications: {', '.join(changes['discontinued_medications'])}")
        
        if changes["changed_values"]:
            for change in changes["changed_values"]:
                summary_parts.append(f"{change['indicator']} changed from {change['previous']} to {change['current']}")
        
        if changes["stable_conditions"]:
            summary_parts.append(f"Stable conditions: {', '.join(changes['stable_conditions'])}")
        
        if len(summary_parts) == 1:  # Only the header
            summary_parts.append("No significant changes detected.")
        
        return "\n".join(summary_parts)

# Modify HealthAnalysisAgent to include episodic memory
class HealthAnalysisAgent:
    def __init__(self, custom_config: Optional[Config] = None):
        # ... existing initialization ...
        
        # Add episodic memory manager
        self.episodic_memory = EpisodicMemoryManager()
        
        logger.info("Episodic memory system initialized")

    # Add new node to LangGraph workflow
    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with episodic memory"""
        logger.info("Setting up Enhanced LangGraph workflow with episodic memory...")

        workflow = StateGraph(HealthAnalysisState)

        # Add all existing nodes...
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("extract_memidnum", self.extract_memidnum)  # New node
        workflow.add_node("deidentify_claims_data", self.deidentify_claims_data)
        workflow.add_node("extract_claims_fields", self.extract_claims_fields)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("compare_with_previous", self.compare_with_previous)  # New node
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("predict_heart_attack", self.predict_heart_attack)
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)
        workflow.add_node("handle_error", self.handle_error)

        # Modify workflow edges to include new nodes
        workflow.add_edge(START, "fetch_api_data")
        
        workflow.add_conditional_edges(
            "fetch_api_data",
            self.should_continue_after_api,
            {
                "continue": "extract_memidnum",
                "retry": "fetch_api_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("extract_memidnum", "deidentify_claims_data")
        # ... continue with existing edges but insert compare_with_previous after extract_entities
        
        workflow.add_edge("extract_entities", "compare_with_previous")
        workflow.add_edge("compare_with_previous", "analyze_trajectory")
        
        # ... rest of existing edges ...

        # Compile with checkpointer
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)

    # New LangGraph node
    def extract_memidnum(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node: Extract memidnum from mcid data"""
        logger.info("Extracting memidnum from mcid data...")
        state["current_step"] = "extract_memidnum"
        
        try:
            mcid_data = state.get("mcid_output", {})
            memidnum = self.episodic_memory.extract_memidnum(mcid_data)
            
            if memidnum:
                state["memidnum"] = memidnum
                logger.info(f"Extracted memidnum: {memidnum}")
            else:
                state["memidnum"] = ""
                logger.warning("Could not extract memidnum from mcid data")
                
        except Exception as e:
            error_msg = f"Error extracting memidnum: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state

    # New LangGraph node
    def compare_with_previous(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node: Compare current entities with previous episodes"""
        logger.info("Comparing with previous health entities...")
        state["current_step"] = "compare_with_previous"
        
        try:
            memidnum = state.get("memidnum", "")
            current_entities = state.get("entity_extraction", {})
            
            if not memidnum:
                state["health_entity_comparison"] = {"no_memidnum": True}
                logger.info("No memidnum available for comparison")
                return state
            
            # Get previous entities
            previous_episode = self.episodic_memory.get_previous_entities(memidnum)
            
            if previous_episode:
                # Perform comparison
                comparison = self.episodic_memory.compare_health_entities(current_entities, previous_episode)
                state["health_entity_comparison"] = comparison
                
                logger.info(f"Health entity comparison completed for memidnum: {memidnum}")
                logger.info(f"Changes detected: {len(comparison.get('changes', {}).get('new_conditions', []))} new conditions")
            else:
                state["health_entity_comparison"] = {"first_episode": True}
                logger.info(f"First episode for memidnum: {memidnum}")
            
            # Store current entities for future comparisons
            self.episodic_memory.store_health_entities(memidnum, current_entities)
            
        except Exception as e:
            error_msg = f"Error in episodic comparison: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state

    # Modify the chatbot to include temporal analysis
    def _handle_general_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle general questions with episodic memory context"""
        try:
            # Get episodic comparison data
            health_comparison = chat_context.get("health_entity_comparison", {})
            memidnum = chat_context.get("memidnum", "")
            
            # Check if this is a temporal/comparison question
            temporal_keywords = [
                'change', 'different', 'compare', 'previous', 'before', 'then vs now', 
                'improvement', 'worse', 'better', 'progress', 'evolution', 'timeline',
                'last time', 'previously', 'history'
            ]
            
            is_temporal_query = any(keyword in user_query.lower() for keyword in temporal_keywords)
            
            # Prepare context with episodic memory
            complete_context = self.data_processor.prepare_enhanced_clinical_context(chat_context)
            
            # Add episodic memory context if available
            episodic_context = ""
            if health_comparison and not health_comparison.get("no_memidnum") and not health_comparison.get("first_episode"):
                episodic_context = f"""
**EPISODIC MEMORY - HEALTH CHANGES ANALYSIS:**
Member ID: {memidnum}
{json.dumps(health_comparison, indent=2)}

**TEMPORAL ANALYSIS INSTRUCTIONS:**
- When discussing health status, reference both current state and changes over time
- Highlight new conditions, resolved conditions, and medication changes
- Explain clinical significance of observed changes
- Provide insights into disease progression or improvement
"""

            # Build conversation history
            history_text = "No previous conversation"
            if chat_history:
                recent_history = chat_history[-5:]
                history_lines = []
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    content = msg['content'][:150]
                    history_lines.append(f"{role}: {content}")
                history_text = "\n".join(history_lines)

            # Create comprehensive analysis prompt with temporal context
            comprehensive_prompt = f"""You are Dr. TemporalAI, a healthcare data analyst specializing in longitudinal health analysis with episodic memory capabilities.

**COMPREHENSIVE DATA ACCESS:**
{complete_context}

{episodic_context}

**CONVERSATION HISTORY:**
{history_text}

**PATIENT QUESTION:** {user_query}

**ANALYSIS INSTRUCTIONS:**

🕒 **TEMPORAL ANALYSIS PRIORITY:**
{"- This appears to be a temporal/comparative question - prioritize episodic memory analysis" if is_temporal_query else "- Standard analysis with temporal context when relevant"}
- Always reference both current state and historical changes when available
- Explain the clinical significance of any observed changes over time
- Highlight trends in health status, medication management, and condition progression

**RESPONSE REQUIREMENTS:**
- Use comprehensive claims data and episodic memory for temporal insights
- Reference specific changes in conditions, medications, and health indicators
- Explain whether changes represent improvement, deterioration, or normal variation
- Provide evidence-based interpretation of health progression
- Include specific dates and timeframes when available

**COMPREHENSIVE RESPONSE WITH TEMPORAL CONTEXT:**
[Provide detailed analysis incorporating both current state and historical changes]"""

            logger.info(f"Processing temporal query: {user_query[:50]}...")

            response = self.api_integrator.call_llm_enhanced(comprehensive_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return "I encountered an error processing your question with temporal context. Please try rephrasing it."

            return response

        except Exception as e:
            logger.error(f"Error in temporal question handling: {str(e)}")
            return "I encountered an error with temporal analysis. Please try again."
