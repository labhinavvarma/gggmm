# episodic_memory.py
# Episodic Memory System for Healthcare Data Analysis

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import uuid
import logging

logger = logging.getLogger(__name__)

class EpisodicMemoryManager:
    """
    Episodic Memory Manager for healthcare data - stores and analyzes temporal health events
    """
    
    def __init__(self, max_episodes_per_patient: int = 1000, retention_days: int = 365*5):
        """
        Initialize episodic memory manager
        
        Args:
            max_episodes_per_patient: Maximum episodes to store per patient
            retention_days: Days to retain episodes before archiving
        """
        self.episodes = {}  # patient_id -> list of episodes
        self.episode_index = {}  # episode_id -> episode
        self.temporal_index = {}  # patient_id -> sorted list of (timestamp, episode_id)
        self.condition_index = defaultdict(list)  # condition -> list of episode_ids
        self.medication_index = defaultdict(list)  # medication -> list of episode_ids
        
        self.max_episodes_per_patient = max_episodes_per_patient
        self.retention_days = retention_days
        
        logger.info(f"🧠 Episodic Memory Manager initialized: {max_episodes_per_patient} max episodes, {retention_days} day retention")

    def create_episode(self, patient_id: str, entity_extraction: Dict[str, Any], 
                      mcid_data: Dict[str, Any] = None, 
                      additional_context: Dict[str, Any] = None) -> str:
        """
        Create a new health episode from entity extraction and MCID data
        
        Args:
            patient_id: Unique patient identifier
            entity_extraction: Results from extract_health_entities_with_clinical_insights
            mcid_data: MCID claims data
            additional_context: Additional contextual information
            
        Returns:
            episode_id: Unique identifier for the created episode
        """
        try:
            episode_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create structured episode
            episode = {
                "episode_id": episode_id,
                "patient_id": patient_id,
                "timestamp": timestamp.isoformat(),
                "timestamp_unix": timestamp.timestamp(),
                
                # Core health state snapshot
                "health_state": {
                    "diabetics": entity_extraction.get("diabetics", "no"),
                    "age_group": entity_extraction.get("age_group", "unknown"),
                    "age": entity_extraction.get("age"),
                    "smoking": entity_extraction.get("smoking", "no"),
                    "alcohol": entity_extraction.get("alcohol", "no"),
                    "blood_pressure": entity_extraction.get("blood_pressure", "unknown")
                },
                
                # Medical conditions and medications
                "medical_conditions": entity_extraction.get("medical_conditions", []),
                "medications_identified": entity_extraction.get("medications_identified", []),
                
                # Analysis metadata
                "analysis_metadata": {
                    "stable_analysis": entity_extraction.get("stable_analysis", False),
                    "llm_analysis": entity_extraction.get("llm_analysis", "not_performed"),
                    "analysis_details": entity_extraction.get("analysis_details", [])
                },
                
                # MCID data context
                "mcid_context": self._process_mcid_context(mcid_data) if mcid_data else None,
                
                # Additional context
                "additional_context": additional_context or {},
                
                # Episode metadata
                "created_at": timestamp.isoformat(),
                "version": "1.0",
                "memory_source": "health_data_processor"
            }
            
            # Store episode and update indices
            self._store_episode(episode)
            self._update_indices(episode)
            
            logger.info(f"🧠 Created episode {episode_id} for patient {patient_id}")
            return episode_id
            
        except Exception as e:
            logger.error(f"❌ Error creating episode for patient {patient_id}: {e}")
            raise

    def _process_mcid_context(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCID data into structured context for episodic storage"""
        try:
            if not mcid_data:
                return None
                
            mcid_context = {
                "data_type": mcid_data.get("data_type", "mcid_claims"),
                "processing_method": mcid_data.get("processing_method", "unknown"),
                "deidentification_timestamp": mcid_data.get("deidentification_timestamp"),
                "claims_summary": {},
                "processing_metadata": {
                    "original_structure_preserved": mcid_data.get("original_structure_preserved", False),
                    "error": mcid_data.get("error")
                }
            }
            
            # Extract and summarize claims data
            claims_data = mcid_data.get("mcid_claims_data", {})
            if claims_data and not mcid_data.get("error"):
                mcid_context["claims_summary"] = {
                    "total_records": self._count_records_recursive(claims_data),
                    "data_structure_type": type(claims_data).__name__,
                    "has_nested_data": self._has_nested_structures(claims_data)
                }
            
            return mcid_context
            
        except Exception as e:
            logger.error(f"❌ Error processing MCID context: {e}")
            return {"processing_error": str(e)}

    def _count_records_recursive(self, data: Any, count: int = 0) -> int:
        """Count total records in nested data structure"""
        if isinstance(data, dict):
            count += 1
            for value in data.values():
                count = self._count_records_recursive(value, count)
        elif isinstance(data, list):
            count += len(data)
            for item in data:
                count = self._count_records_recursive(item, count)
        return count

    def _has_nested_structures(self, data: Any) -> bool:
        """Check if data contains nested dict/list structures"""
        if isinstance(data, dict):
            return any(isinstance(v, (dict, list)) for v in data.values())
        elif isinstance(data, list):
            return any(isinstance(item, (dict, list)) for item in data)
        return False

    def _store_episode(self, episode: Dict[str, Any]):
        """Store episode in memory structures with proper indexing"""
        patient_id = episode["patient_id"]
        episode_id = episode["episode_id"]
        
        # Initialize patient storage if needed
        if patient_id not in self.episodes:
            self.episodes[patient_id] = deque(maxlen=self.max_episodes_per_patient)
            self.temporal_index[patient_id] = []
        
        # Store episode
        self.episodes[patient_id].append(episode)
        self.episode_index[episode_id] = episode
        
        # Update temporal index for chronological access
        timestamp = episode["timestamp_unix"]
        self.temporal_index[patient_id].append((timestamp, episode_id))
        self.temporal_index[patient_id].sort(key=lambda x: x[0])
        
        logger.debug(f"🧠 Stored episode {episode_id} for patient {patient_id}")

    def _update_indices(self, episode: Dict[str, Any]):
        """Update searchable indices for fast retrieval"""
        episode_id = episode["episode_id"]
        
        # Index medical conditions for search
        for condition in episode["medical_conditions"]:
            if condition and isinstance(condition, str):
                condition_key = condition.lower().strip()
                self.condition_index[condition_key].append(episode_id)
        
        # Index medications for search
        for med_info in episode["medications_identified"]:
            if isinstance(med_info, dict):
                med_name = med_info.get("label_name", "")
                if med_name and isinstance(med_name, str):
                    med_key = med_name.lower().strip()
                    self.medication_index[med_key].append(episode_id)
        
        logger.debug(f"🧠 Updated indices for episode {episode_id}")

    def get_patient_timeline(self, patient_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of episodes for a patient
        
        Args:
            patient_id: Patient identifier
            limit: Maximum episodes to return (most recent first)
            
        Returns:
            List of episodes ordered by timestamp (newest first)
        """
        try:
            if patient_id not in self.episodes:
                logger.warning(f"🧠 No episodes found for patient {patient_id}")
                return []
            
            # Get episodes sorted by timestamp (newest first)
            patient_episodes = list(self.episodes[patient_id])
            patient_episodes.sort(key=lambda x: x["timestamp_unix"], reverse=True)
            
            if limit and limit > 0:
                patient_episodes = patient_episodes[:limit]
            
            logger.info(f"🧠 Retrieved {len(patient_episodes)} episodes for patient {patient_id}")
            return patient_episodes
            
        except Exception as e:
            logger.error(f"❌ Error retrieving timeline for patient {patient_id}: {e}")
            return []

    def analyze_health_progression(self, patient_id: str, 
                                 analysis_window_days: int = 90) -> Dict[str, Any]:
        """
        Analyze health progression patterns for a patient over time
        
        Args:
            patient_id: Patient identifier
            analysis_window_days: Days to analyze (looking back from now)
            
        Returns:
            Comprehensive health progression analysis
        """
        try:
            logger.info(f"🧠 Analyzing health progression for patient {patient_id} over {analysis_window_days} days")
            
            timeline = self.get_patient_timeline(patient_id)
            if not timeline:
                return {"error": f"No episodes found for patient {patient_id}"}
            
            # Filter episodes within analysis window
            cutoff_time = datetime.now() - timedelta(days=analysis_window_days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            recent_episodes = [
                ep for ep in timeline 
                if ep["timestamp_unix"] >= cutoff_timestamp
            ]
            
            if not recent_episodes:
                return {
                    "error": f"No episodes found within {analysis_window_days} day window",
                    "total_episodes": len(timeline),
                    "oldest_episode_date": timeline[-1]["timestamp"] if timeline else None
                }
            
            # Perform comprehensive analysis
            analysis = {
                "patient_id": patient_id,
                "analysis_window_days": analysis_window_days,
                "episodes_analyzed": len(recent_episodes),
                "total_episodes": len(timeline),
                "date_range": {
                    "start": recent_episodes[-1]["timestamp"],  # Oldest in window
                    "end": recent_episodes[0]["timestamp"]      # Most recent
                },
                
                # Core analyses
                "health_state_progression": self._analyze_health_state_changes(recent_episodes),
                "medication_changes": self._analyze_medication_timeline(recent_episodes),
                "condition_evolution": self._analyze_condition_progression(recent_episodes),
                "risk_factor_analysis": self._analyze_risk_factors(recent_episodes),
                
                # Summary insights
                "key_insights": self._generate_progression_insights(recent_episodes),
                "stability_metrics": self._calculate_stability_metrics(recent_episodes)
            }
            
            logger.info(f"✅ Health progression analysis completed for patient {patient_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing health progression for {patient_id}: {e}")
            return {"error": str(e), "patient_id": patient_id}

    def _analyze_health_state_changes(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze changes in core health states over the episode timeline"""
        state_keys = ["diabetics", "smoking", "alcohol", "blood_pressure"]
        changes_detected = {}
        current_states = {}
        
        # Track changes for each health state
        for state_key in state_keys:
            changes = []
            previous_value = None
            
            # Process episodes in chronological order (oldest first)
            for episode in reversed(episodes):
                current_value = episode["health_state"].get(state_key, "unknown")
                
                if previous_value is not None and current_value != previous_value:
                    changes.append({
                        "timestamp": episode["timestamp"],
                        "change_from": previous_value,
                        "change_to": current_value,
                        "episode_id": episode["episode_id"]
                    })
                
                previous_value = current_value
            
            changes_detected[state_key] = changes
            current_states[state_key] = previous_value
        
        return {
            "state_changes": changes_detected,
            "current_states": current_states,
            "change_summary": self._summarize_state_changes(changes_detected)
        }

    def _analyze_medication_timeline(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze medication additions, discontinuations, and patterns"""
        medication_events = []
        all_medications_seen = set()
        current_medications = set()
        
        # Process in chronological order
        for episode in reversed(episodes):
            episode_meds = {
                med["label_name"] for med in episode["medications_identified"] 
                if med.get("label_name")
            }
            
            # Detect changes
            added_meds = episode_meds - current_medications
            removed_meds = current_medications - episode_meds
            
            if added_meds or removed_meds:
                medication_events.append({
                    "timestamp": episode["timestamp"],
                    "added": list(added_meds),
                    "removed": list(removed_meds),
                    "total_medications": len(episode_meds),
                    "episode_id": episode["episode_id"]
                })
            
            current_medications = episode_meds
            all_medications_seen.update(episode_meds)
        
        return {
            "medication_events": medication_events,
            "all_medications": list(all_medications_seen),
            "current_medications": list(current_medications),
            "medication_stability": len(medication_events) == 0,
            "polypharmacy_episodes": sum(1 for ep in episodes if len(ep["medications_identified"]) >= 5)
        }

    def _analyze_condition_progression(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track medical condition changes and progression patterns"""
        condition_timeline = []
        all_conditions = set()
        
        for episode in reversed(episodes):  # Chronological order
            conditions = episode["medical_conditions"]
            condition_timeline.append({
                "timestamp": episode["timestamp"],
                "conditions": conditions,
                "condition_count": len(conditions),
                "episode_id": episode["episode_id"]
            })
            all_conditions.update(conditions)
        
        # Calculate progression metrics
        condition_counts = [entry["condition_count"] for entry in condition_timeline]
        
        return {
            "condition_timeline": condition_timeline,
            "all_conditions_seen": list(all_conditions),
            "condition_count_trend": condition_counts,
            "condition_burden": {
                "min_conditions": min(condition_counts) if condition_counts else 0,
                "max_conditions": max(condition_counts) if condition_counts else 0,
                "avg_conditions": sum(condition_counts) / len(condition_counts) if condition_counts else 0
            }
        }

    def _analyze_risk_factors(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive risk factor analysis across episodes"""
        risk_counts = {
            "diabetes_positive": 0,
            "hypertension_present": 0, 
            "smoking_positive": 0,
            "alcohol_positive": 0,
            "polypharmacy": 0,
            "multiple_conditions": 0
        }
        
        total_episodes = len(episodes)
        
        for episode in episodes:
            health_state = episode["health_state"]
            
            # Count risk factors
            if health_state.get("diabetics") == "yes":
                risk_counts["diabetes_positive"] += 1
            if health_state.get("blood_pressure") in ["diagnosed", "managed"]:
                risk_counts["hypertension_present"] += 1
            if health_state.get("smoking") == "yes":
                risk_counts["smoking_positive"] += 1
            if health_state.get("alcohol") == "yes":
                risk_counts["alcohol_positive"] += 1
            if len(episode["medications_identified"]) >= 5:
                risk_counts["polypharmacy"] += 1
            if len(episode["medical_conditions"]) >= 3:
                risk_counts["multiple_conditions"] += 1
        
        # Calculate percentages
        risk_percentages = {
            f"{key}_percentage": (count / total_episodes * 100) if total_episodes > 0 else 0
            for key, count in risk_counts.items()
        }
        
        return {
            "risk_factor_counts": risk_counts,
            "risk_factor_percentages": risk_percentages,
            "total_episodes_analyzed": total_episodes,
            "high_risk_episodes": sum(1 for ep in episodes if self._is_high_risk_episode(ep))
        }

    def _is_high_risk_episode(self, episode: Dict[str, Any]) -> bool:
        """Determine if an episode represents high cardiovascular risk"""
        health_state = episode["health_state"]
        
        risk_factors = [
            health_state.get("diabetics") == "yes",
            health_state.get("blood_pressure") in ["diagnosed", "managed"],
            health_state.get("smoking") == "yes",
            len(episode["medications_identified"]) >= 5,
            len(episode["medical_conditions"]) >= 3
        ]
        
        return sum(risk_factors) >= 2  # 2 or more risk factors = high risk

    def _generate_progression_insights(self, episodes: List[Dict[str, Any]]) -> List[str]:
        """Generate human-readable insights about health progression"""
        insights = []
        
        if len(episodes) < 2:
            insights.append("Insufficient episodes for progression analysis")
            return insights
        
        # Compare first vs last episode
        oldest = episodes[-1]
        newest = episodes[0]
        
        # Diabetes progression
        if oldest["health_state"]["diabetics"] != newest["health_state"]["diabetics"]:
            insights.append(f"Diabetes status changed from {oldest['health_state']['diabetics']} to {newest['health_state']['diabetics']}")
        
        # Medication changes
        old_med_count = len(oldest["medications_identified"])
        new_med_count = len(newest["medications_identified"])
        if abs(old_med_count - new_med_count) >= 2:
            change_type = "increased" if new_med_count > old_med_count else "decreased"
            insights.append(f"Medication count {change_type} from {old_med_count} to {new_med_count}")
        
        # Condition progression
        old_condition_count = len(oldest["medical_conditions"])
        new_condition_count = len(newest["medical_conditions"])
        if new_condition_count > old_condition_count:
            insights.append(f"New medical conditions identified ({old_condition_count} → {new_condition_count})")
        
        if not insights:
            insights.append("Health profile appears stable over analysis period")
            
        return insights

    def _calculate_stability_metrics(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics indicating health stability"""
        if len(episodes) < 2:
            return {"insufficient_data": True}
        
        # Health state stability
        state_changes = 0
        state_keys = ["diabetics", "smoking", "alcohol", "blood_pressure"]
        
        for i in range(1, len(episodes)):
            for key in state_keys:
                if episodes[i]["health_state"].get(key) != episodes[i-1]["health_state"].get(key):
                    state_changes += 1
        
        # Medication stability
        med_changes = 0
        for i in range(1, len(episodes)):
            current_meds = {med["label_name"] for med in episodes[i]["medications_identified"]}
            previous_meds = {med["label_name"] for med in episodes[i-1]["medications_identified"]}
            if current_meds != previous_meds:
                med_changes += 1
        
        return {
            "health_state_changes": state_changes,
            "medication_changes": med_changes,
            "stability_score": max(0, 100 - (state_changes + med_changes) * 10),  # 0-100 scale
            "episodes_analyzed": len(episodes)
        }

    def _summarize_state_changes(self, changes: Dict[str, List]) -> List[str]:
        """Create summary of health state changes"""
        summary = []
        
        for state, change_list in changes.items():
            if change_list:
                summary.append(f"{state.replace('_', ' ').title()}: {len(change_list)} changes detected")
        
        if not summary:
            summary.append("No significant health state changes detected")
            
        return summary

    def search_episodes_by_condition(self, condition_pattern: str) -> List[Dict[str, Any]]:
        """Search episodes containing specific medical conditions"""
        matching_episodes = []
        pattern_lower = condition_pattern.lower()
        
        for condition_key, episode_ids in self.condition_index.items():
            if pattern_lower in condition_key:
                for episode_id in episode_ids:
                    if episode_id in self.episode_index:
                        matching_episodes.append(self.episode_index[episode_id])
        
        # Remove duplicates and sort by timestamp
        unique_episodes = {ep["episode_id"]: ep for ep in matching_episodes}
        sorted_episodes = sorted(unique_episodes.values(), 
                               key=lambda x: x["timestamp_unix"], reverse=True)
        
        logger.info(f"🧠 Found {len(sorted_episodes)} episodes matching condition pattern: {condition_pattern}")
        return sorted_episodes

    def search_episodes_by_medication(self, medication_pattern: str) -> List[Dict[str, Any]]:
        """Search episodes containing specific medications"""
        matching_episodes = []
        pattern_lower = medication_pattern.lower()
        
        for med_key, episode_ids in self.medication_index.items():
            if pattern_lower in med_key:
                for episode_id in episode_ids:
                    if episode_id in self.episode_index:
                        matching_episodes.append(self.episode_index[episode_id])
        
        # Remove duplicates and sort by timestamp
        unique_episodes = {ep["episode_id"]: ep for ep in matching_episodes}
        sorted_episodes = sorted(unique_episodes.values(), 
                               key=lambda x: x["timestamp_unix"], reverse=True)
        
        logger.info(f"🧠 Found {len(sorted_episodes)} episodes matching medication pattern: {medication_pattern}")
        return sorted_episodes

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Comprehensive episodic memory statistics"""
        total_episodes = len(self.episode_index)
        total_patients = len(self.episodes)
        
        # Patient episode distribution
        episode_counts = [len(episodes) for episodes in self.episodes.values()]
        avg_episodes = sum(episode_counts) / len(episode_counts) if episode_counts else 0
        
        # Date range analysis
        timestamps = [ep["timestamp_unix"] for ep in self.episode_index.values()]
        date_range = {}
        if timestamps:
            earliest = datetime.fromtimestamp(min(timestamps))
            latest = datetime.fromtimestamp(max(timestamps))
            date_range = {
                "earliest_episode": earliest.isoformat(),
                "latest_episode": latest.isoformat(),
                "total_time_span_days": (latest - earliest).days
            }
        
        return {
            "memory_overview": {
                "total_episodes": total_episodes,
                "total_patients": total_patients,
                "average_episodes_per_patient": round(avg_episodes, 2),
                "max_episodes_per_patient": max(episode_counts) if episode_counts else 0,
                "min_episodes_per_patient": min(episode_counts) if episode_counts else 0
            },
            
            "temporal_coverage": date_range,
            
            "indexing_statistics": {
                "indexed_conditions": len(self.condition_index),
                "indexed_medications": len(self.medication_index),
                "total_condition_references": sum(len(episodes) for episodes in self.condition_index.values()),
                "total_medication_references": sum(len(episodes) for episodes in self.medication_index.values())
            },
            
            "configuration": {
                "max_episodes_per_patient": self.max_episodes_per_patient,
                "retention_days": self.retention_days
            }
        }

    def export_patient_data(self, patient_id: str, include_analysis: bool = True) -> Dict[str, Any]:
        """Export complete episodic data for a specific patient"""
        try:
            timeline = self.get_patient_timeline(patient_id)
            
            export_data = {
                "patient_id": patient_id,
                "export_timestamp": datetime.now().isoformat(),
                "episode_count": len(timeline),
                "episodes": timeline
            }
            
            if include_analysis and timeline:
                export_data["health_analysis"] = self.analyze_health_progression(
                    patient_id, 
                    analysis_window_days=365  # Full year analysis
                )
            
            logger.info(f"🧠 Exported {len(timeline)} episodes for patient {patient_id}")
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Error exporting data for patient {patient_id}: {e}")
            return {"error": str(e), "patient_id": patient_id}

    def cleanup_old_episodes(self, days_to_retain: int = None) -> Dict[str, Any]:
        """Remove episodes older than retention period"""
        retention_days = days_to_retain or self.retention_days
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        cutoff_timestamp = cutoff_time.timestamp()
        
        cleanup_stats = {
            "episodes_before": len(self.episode_index),
            "patients_before": len(self.episodes),
            "episodes_removed": 0,
            "patients_affected": 0
        }
        
        try:
            episodes_to_remove = []
            
            # Find old episodes
            for episode_id, episode in self.episode_index.items():
                if episode["timestamp_unix"] < cutoff_timestamp:
                    episodes_to_remove.append(episode_id)
            
            # Remove old episodes
            for episode_id in episodes_to_remove:
                episode = self.episode_index[episode_id]
                patient_id = episode["patient_id"]
                
                # Remove from main storage
                del self.episode_index[episode_id]
                
                # Remove from patient episodes
                if patient_id in self.episodes:
                    self.episodes[patient_id] = deque(
                        [ep for ep in self.episodes[patient_id] if ep["episode_id"] != episode_id],
                        maxlen=self.max_episodes_per_patient
                    )
                
                # Update temporal index
                if patient_id in self.temporal_index:
                    self.temporal_index[patient_id] = [
                        (ts, eid) for ts, eid in self.temporal_index[patient_id] 
                        if eid != episode_id
                    ]
                
                cleanup_stats["episodes_removed"] += 1
            
            # Rebuild search indices
            self._rebuild_search_indices()
            
            cleanup_stats.update({
                "episodes_after": len(self.episode_index),
                "patients_after": len(self.episodes),
                "cleanup_completed": True
            })
            
            logger.info(f"🧠 Cleanup completed: removed {cleanup_stats['episodes_removed']} old episodes")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"❌ Error during episode cleanup: {e}")
            cleanup_stats["error"] = str(e)
            return cleanup_stats

    def _rebuild_search_indices(self):
        """Rebuild condition and medication search indices"""
        self.condition_index.clear()
        self.medication_index.clear()
        
        for episode in self.episode_index.values():
            self._update_indices(episode)
        
        logger.info("🧠 Search indices rebuilt")
