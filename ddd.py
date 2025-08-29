"""
Final Episodic Memory Manager with Simplified Structure

This implementation stores only mcid, entity_extraction, and timestamp
as requested, with append functionality for returning patients.

SECURITY WARNING: This stores healthcare data in unencrypted local files.
Do not use in production healthcare environments.
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import shutil
import time

logger = logging.getLogger(__name__)

@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory storage"""
    storage_directory: str = "./episodic_memory"
    file_prefix: str = "patient_memory"
    backup_directory: str = "./episodic_memory_backup"
    retention_days: int = 365

class EpisodicMemoryError(Exception):
    """Custom exception for episodic memory operations"""
    pass

class EpisodicMemoryManager:
    """
    Manages episodic memory for patient healthcare data with simplified structure
    
    JSON Structure:
    Single visit: {"id_type": "mcid", "mcid": "123", "entity_extraction": {...}, "timestamp": "..."}
    Multiple visits: [visit1, visit2, visit3, ...]
    """
    
    def __init__(self, config: EpisodicMemoryConfig):
        self.config = config
        self.storage_path = Path(config.storage_directory)
        self.backup_path = Path(config.backup_directory)
        
        # Ensure directories exist
        self._setup_directories()
        
        logger.info("EpisodicMemoryManager initialized with simplified structure")
    
    def _setup_directories(self):
        """Setup storage and backup directories"""
        try:
            self.storage_path.mkdir(exist_ok=True, parents=True)
            self.backup_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"Storage directories created: {self.storage_path}, {self.backup_path}")
        except Exception as e:
            raise EpisodicMemoryError(f"Failed to setup directories: {e}")
    
    def _extract_mcid_list(self, deidentified_mcid: Dict[str, Any]) -> str:
        """Extract MCID from 'mcidList' attribute in body section"""
        try:
            # Navigate to the body section first
            body = deidentified_mcid.get("body", {})
            
            if not body:
                logger.warning("No 'body' section found in deidentified_mcid")
                return ""
            
            # Look for mcidList in the body section
            mcid_value = body.get("mcidList", "")
            
            if mcid_value and str(mcid_value).strip():
                mcid_str = str(mcid_value).strip()
                logger.info(f"Found mcidList: {mcid_str}")
                return mcid_str
            else:
                logger.warning("No 'mcidList' value found in body section")
                return ""
            
        except Exception as e:
            logger.error(f"Error extracting mcidList: {e}")
            return ""
    
    def _generate_patient_id_hash(self, mcid: str) -> str:
        """Generate a secure hash from MCID for filename"""
        if not mcid:
            raise EpisodicMemoryError("Cannot generate hash from empty MCID")
        
        # Create SHA256 hash
        hash_object = hashlib.sha256(mcid.encode('utf-8'))
        return hash_object.hexdigest()[:16]  # Use first 16 characters
    
    def _get_memory_file_path(self, patient_hash: str) -> Path:
        """Get the file path for patient's episodic memory"""
        filename = f"{self.config.file_prefix}_{patient_hash}.json"
        return self.storage_path / filename
    
    def _get_backup_file_path(self, patient_hash: str) -> Path:
        """Get the backup file path for patient's episodic memory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.file_prefix}_{patient_hash}_{timestamp}.backup"
        return self.backup_path / filename
    
    @contextmanager
    def _file_lock(self, file_path: Path, mode: str = "r"):
        """Context manager for file access (Windows compatible - no locking)"""
        max_attempts = 5
        delay = 0.1
        
        for attempt in range(max_attempts):
            try:
                with open(file_path, mode, encoding='utf-8') as f:
                    yield f
                return
                    
            except PermissionError:
                if attempt < max_attempts - 1:
                    logger.warning(f"File access blocked, retrying in {delay}s... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"File access failed after {max_attempts} attempts")
                    raise
            except FileNotFoundError:
                if 'r' in mode:
                    yield None
                    return
                else:
                    raise
    
    def _load_existing_memory(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load existing episodic memory file"""
        try:
            if not file_path.exists():
                return None
            
            with self._file_lock(file_path, 'r') as f:
                if f is None:
                    return None
                
                data = json.load(f)
                
                # Validate basic structure
                if not isinstance(data, (dict, list)):
                    logger.error(f"Invalid memory file format: {file_path}")
                    return None
                
                return data
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in memory file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading memory file {file_path}: {e}")
            return None
    
    def _create_backup(self, file_path: Path, patient_hash: str) -> bool:
        """Create a backup of existing memory file"""
        try:
            if not file_path.exists():
                return True
            
            backup_path = self._get_backup_file_path(patient_hash)
            shutil.copy2(file_path, backup_path)
            
            logger.debug(f"Created backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def _save_memory(self, file_path: Path, memory_data: Dict[str, Any], patient_hash: str) -> bool:
        """Save episodic memory data with backup"""
        try:
            # Create backup of existing file
            self._create_backup(file_path, patient_hash)
            
            # Write new data
            with self._file_lock(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"Successfully saved episodic memory: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory file {file_path}: {e}")
            return False
    
    def save_episodic_memory(self, deidentified_mcid: Dict[str, Any], 
                           entity_extraction: Dict[str, Any],
                           additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save or update episodic memory for a patient with simplified structure
        
        Args:
            deidentified_mcid: Deidentified MCID data
            entity_extraction: Extracted health entities
            additional_metadata: Optional additional metadata (ignored - for compatibility)
        
        Returns:
            Dict containing operation result and memory info
        """
        try:
            # Extract MCID from 'mcidList' attribute only
            mcid = self._extract_mcid_list(deidentified_mcid)
            
            if not mcid:
                return {
                    "success": False,
                    "error": "No 'mcidList' found in deidentified data",
                    "mcid": "",
                    "operation": "none"
                }
            
            # Generate patient hash for filename
            patient_hash = self._generate_patient_id_hash(mcid)
            file_path = self._get_memory_file_path(patient_hash)
            
            # Create simplified entity extraction with only required fields
            simplified_entities = {
                "diabetics": entity_extraction.get("diabetics", "unknown"),
                "blood_pressure": entity_extraction.get("blood_pressure", "unknown"),
                "age": entity_extraction.get("age", "unknown"),
                "smoking": entity_extraction.get("smoking", "unknown"),
                "alcohol": entity_extraction.get("alcohol", "unknown")
            }
            
            # Create new visit entry with exact structure requested
            current_timestamp = datetime.now().isoformat()
            new_visit = {
                "id_type": "mcid",
                "mcid": mcid,
                "entity_extraction": simplified_entities,
                "timestamp": current_timestamp
            }
            
            # Check if file exists for returning patient
            existing_memory = self._load_existing_memory(file_path)
            
            if existing_memory:
                # Returning patient - append to existing data
                operation = "updated"
                
                # If existing data is not a list, convert it
                if isinstance(existing_memory, dict) and "id_type" in existing_memory:
                    # Old single visit format, convert to list
                    memory_data = [existing_memory, new_visit]
                elif isinstance(existing_memory, list):
                    # Already a list of visits, append new visit
                    memory_data = existing_memory + [new_visit]
                else:
                    # Unknown format, start fresh
                    memory_data = [new_visit]
            else:
                # New patient - single visit
                operation = "created"
                memory_data = new_visit
            
            # Save memory data
            save_success = self._save_memory(file_path, memory_data, patient_hash)
            
            # Calculate visit count
            visit_count = len(memory_data) if isinstance(memory_data, list) else 1
            
            return {
                "success": save_success,
                "operation": operation,
                "patient_hash": patient_hash,
                "file_path": str(file_path),
                "mcid": mcid,
                "visit_count": visit_count,
                "timestamp": current_timestamp
            }
            
        except Exception as e:
            logger.error(f"Error in save_episodic_memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": "error"
            }
    
    def load_episodic_memory(self, mcid: str) -> Optional[Dict[str, Any]]:
        """Load episodic memory for given MCID"""
        try:
            if not mcid:
                return None
            
            patient_hash = self._generate_patient_id_hash(mcid)
            file_path = self._get_memory_file_path(patient_hash)
            
            return self._load_existing_memory(file_path)
            
        except Exception as e:
            logger.error(f"Error loading episodic memory: {e}")
            return None
    
    def get_patient_history(self, mcid: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get patient's visit history"""
        memory = self.load_episodic_memory(mcid)
        if not memory:
            return []
        
        # Handle both single visit and multiple visits format
        if isinstance(memory, list):
            # Multiple visits (returning patient)
            history = memory
        elif isinstance(memory, dict) and "id_type" in memory:
            # Single visit (first-time patient)
            history = [memory]
        else:
            return []
        
        # Apply limit if specified
        if limit and len(history) > limit:
            history = history[-limit:]  # Get most recent visits
        
        return history
    
    def get_patient_summary(self, mcid: str) -> Optional[Dict[str, Any]]:
        """Get a summary of patient's episodic memory"""
        memory = self.load_episodic_memory(mcid)
        if not memory:
            return None
        
        # Handle both single visit and multiple visits format
        if isinstance(memory, list):
            # Multiple visits - get latest
            latest_visit = memory[-1] if memory else {}
            visit_count = len(memory)
            first_visit = memory[0] if memory else {}
        elif isinstance(memory, dict) and "id_type" in memory:
            # Single visit
            latest_visit = memory
            visit_count = 1
            first_visit = memory
        else:
            return None
        
        latest_entities = latest_visit.get('entity_extraction', {})
        
        return {
            "patient_hash": self._generate_patient_id_hash(mcid),
            "mcid": mcid,
            "total_visits": visit_count,
            "first_visit": first_visit.get('timestamp'),
            "last_visit": latest_visit.get('timestamp'),
            "current_diabetics": latest_entities.get('diabetics', 'unknown'),
            "current_blood_pressure": latest_entities.get('blood_pressure', 'unknown'),
            "current_age": latest_entities.get('age', 'unknown'),
            "current_smoking": latest_entities.get('smoking', 'unknown'),
            "current_alcohol": latest_entities.get('alcohol', 'unknown')
        }
    
    def delete_patient_memory(self, mcid: str) -> bool:
        """Delete patient's episodic memory"""
        try:
            if not mcid:
                return False
            
            patient_hash = self._generate_patient_id_hash(mcid)
            file_path = self._get_memory_file_path(patient_hash)
            
            if file_path.exists():
                # Create final backup before deletion
                self._create_backup(file_path, patient_hash)
                
                # Delete the file
                file_path.unlink()
                logger.info(f"Deleted episodic memory for MCID: {mcid}")
                return True
            else:
                logger.warning(f"No memory file found for MCID: {mcid}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting patient memory: {e}")
            return False
    
    def cleanup_old_memories(self, days_old: int = None) -> Dict[str, int]:
        """Clean up episodic memory files older than specified days"""
        if days_old is None:
            days_old = self.config.retention_days
            
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_count = 0
        error_count = 0
        
        try:
            for file_path in self.storage_path.glob(f"{self.config.file_prefix}_*.json"):
                try:
                    memory = self._load_existing_memory(file_path)
                    if memory:
                        # Handle both single visit and multiple visits format
                        if isinstance(memory, list):
                            # Multiple visits - check latest timestamp
                            latest_timestamp_str = memory[-1].get('timestamp', '') if memory else ''
                        elif isinstance(memory, dict):
                            # Single visit
                            latest_timestamp_str = memory.get('timestamp', '')
                        else:
                            continue
                        
                        if latest_timestamp_str:
                            file_timestamp = datetime.fromisoformat(latest_timestamp_str)
                            if file_timestamp < cutoff_date:
                                # Create backup before deletion
                                patient_hash = file_path.stem.replace(f"{self.config.file_prefix}_", "")
                                self._create_backup(file_path, patient_hash)
                                
                                file_path.unlink()
                                removed_count += 1
                                logger.info(f"Removed old memory file: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    error_count += 1
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            error_count += 1
        
        return {
            "removed_count": removed_count,
            "error_count": error_count,
            "cutoff_date": cutoff_date.isoformat()
        }
    
    def list_all_patients(self) -> List[Dict[str, Any]]:
        """List all patients in episodic memory"""
        patients = []
        
        try:
            for file_path in self.storage_path.glob(f"{self.config.file_prefix}_*.json"):
                try:
                    memory = self._load_existing_memory(file_path)
                    if memory:
                        patient_hash = file_path.stem.replace(f"{self.config.file_prefix}_", "")
                        
                        # Handle both single visit and multiple visits format
                        if isinstance(memory, list):
                            # Multiple visits
                            visit_count = len(memory)
                            latest_visit = memory[-1] if memory else {}
                            mcid = latest_visit.get('mcid', 'unknown')
                            last_updated = latest_visit.get('timestamp')
                        elif isinstance(memory, dict):
                            # Single visit
                            visit_count = 1
                            mcid = memory.get('mcid', 'unknown')
                            last_updated = memory.get('timestamp')
                        else:
                            continue
                        
                        patients.append({
                            "patient_hash": patient_hash,
                            "mcid": mcid,
                            "visit_count": visit_count,
                            "last_updated": last_updated,
                            "file_path": str(file_path)
                        })
                except Exception as e:
                    logger.error(f"Error reading patient file {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error listing patients: {e}")
        
        return sorted(patients, key=lambda x: x.get('last_updated', ''), reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        try:
            patients = self.list_all_patients()
            
            if not patients:
                return {
                    "total_patients": 0,
                    "total_memory_files": 0,
                    "total_visits": 0,
                    "storage_directory": str(self.storage_path),
                    "backup_directory": str(self.backup_path)
                }
            
            total_visits = sum(p.get('visit_count', 0) for p in patients)
            
            # Get oldest and newest
            oldest_patient = min(patients, key=lambda x: x.get('last_updated', ''))
            newest_patient = max(patients, key=lambda x: x.get('last_updated', ''))
            
            return {
                "total_patients": len(patients),
                "total_memory_files": len(list(self.storage_path.glob(f"{self.config.file_prefix}_*.json"))),
                "total_visits": total_visits,
                "oldest_patient_updated": oldest_patient.get('last_updated'),
                "newest_patient_updated": newest_patient.get('last_updated'),
                "storage_directory": str(self.storage_path),
                "backup_directory": str(self.backup_path),
                "retention_days": self.config.retention_days
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Example usage
    config = EpisodicMemoryConfig(
        storage_directory="./test_episodic_memory",
        backup_directory="./test_episodic_memory_backup"
    )
    
    manager = EpisodicMemoryManager(config)
    
    # Test with your JSON structure
    test_mcid_data = {
        "status_code": 200,
        "body": {
            "requestID": "1",
            "processStatus": {
                "completed": "true",
                "isMemput": "false",
                "errorCode": "OK",
                "errorText": ""
            },
            "mcidList": "139407292",
            "mem": None,
            "memidnum": "391709711-000002-003324975",
            "matchScore": "155"
        },
        "service": "mcid",
        "timestamp": "2025-08-28T18:14:34.926435",
        "status": "success"
    }
    
    test_entities = {
        "diabetics": "yes",
        "blood_pressure": "managed",
        "age": 45,
        "smoking": "no",
        "alcohol": "unknown"
    }
    
    result = manager.save_episodic_memory(test_mcid_data, test_entities)
    print(f"Save result: {result}")
    
    if result["success"]:
        mcid = result["mcid"]
        memory = manager.load_episodic_memory(mcid)
        print(f"Loaded memory: {json.dumps(memory, indent=2)}")
