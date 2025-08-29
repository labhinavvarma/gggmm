"""
Fixed Episodic Memory Manager - Corrected mcidList extraction

This implementation fixes the mcidList extraction issue and ensures
episodic memory files are properly generated.

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
    Fixed Episodic Memory Manager for patient healthcare data
    
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
        
        logger.info("EpisodicMemoryManager initialized with fixed mcidList extraction")
    
    def _setup_directories(self):
        """Setup storage and backup directories with debug output"""
        try:
            self.storage_path.mkdir(exist_ok=True, parents=True)
            self.backup_path.mkdir(exist_ok=True, parents=True)
            
            # Debug output
            print(f"DEBUG: Storage path created: {self.storage_path}")
            print(f"DEBUG: Storage path exists: {self.storage_path.exists()}")
            print(f"DEBUG: Backup path created: {self.backup_path}")
            print(f"DEBUG: Backup path exists: {self.backup_path.exists()}")
            
            logger.info(f"Storage directories created: {self.storage_path}, {self.backup_path}")
        except Exception as e:
            print(f"DEBUG: Directory setup failed: {e}")
            raise EpisodicMemoryError(f"Failed to setup directories: {e}")
    
    def _extract_mcid_list(self, deidentified_mcid: Dict[str, Any]) -> str:
        """
        Fixed method to extract MCID from deidentified data structure
        
        Handles both original API structure and deidentified structure:
        1. Original: {"body": {"mcidList": "139407292"}}
        2. Deidentified: {"mcid_claims_data": {"mcidList": "139407292"}}
        """
        try:
            print(f"DEBUG: Full deidentified_mcid structure: {json.dumps(deidentified_mcid, indent=2)}")
            
            mcid_value = None
            
            # Method 1: Check for original structure (body.mcidList)
            body = deidentified_mcid.get("body", {})
            if body and isinstance(body, dict):
                mcid_value = body.get("mcidList", "")
                if mcid_value:
                    print(f"DEBUG: Found mcidList in body: {mcid_value}")
                    logger.info(f"Found mcidList in body section: {mcid_value}")
                    return str(mcid_value).strip()
            
            # Method 2: Check for deidentified structure (mcid_claims_data.mcidList)
            mcid_claims_data = deidentified_mcid.get("mcid_claims_data", {})
            if mcid_claims_data and isinstance(mcid_claims_data, dict):
                mcid_value = mcid_claims_data.get("mcidList", "")
                if mcid_value:
                    print(f"DEBUG: Found mcidList in mcid_claims_data: {mcid_value}")
                    logger.info(f"Found mcidList in mcid_claims_data: {mcid_value}")
                    return str(mcid_value).strip()
            
            # Method 3: Direct search at root level
            mcid_value = deidentified_mcid.get("mcidList", "")
            if mcid_value:
                print(f"DEBUG: Found mcidList at root level: {mcid_value}")
                logger.info(f"Found mcidList at root level: {mcid_value}")
                return str(mcid_value).strip()
            
            # Method 4: Recursive search through all nested structures
            mcid_found = self._recursive_mcid_search(deidentified_mcid)
            if mcid_found:
                print(f"DEBUG: Found mcidList via recursive search: {mcid_found}")
                logger.info(f"Found mcidList via recursive search: {mcid_found}")
                return str(mcid_found).strip()
            
            print("DEBUG: mcidList not found in any structure")
            logger.warning("No 'mcidList' found in any data structure")
            return ""
            
        except Exception as e:
            print(f"DEBUG: Error extracting mcidList: {e}")
            logger.error(f"Error extracting mcidList: {e}")
            return ""
    
    def _recursive_mcid_search(self, data: Any, path: str = "") -> str:
        """Recursively search for mcidList in nested data structures"""
        try:
            if isinstance(data, dict):
                # Check if mcidList exists at this level
                if "mcidList" in data and data["mcidList"]:
                    mcid_value = str(data["mcidList"]).strip()
                    if mcid_value:
                        print(f"DEBUG: Found mcidList at path '{path}': {mcid_value}")
                        return mcid_value
                
                # Search nested dictionaries
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    result = self._recursive_mcid_search(value, new_path)
                    if result:
                        return result
                        
            elif isinstance(data, list):
                # Search nested lists
                for i, item in enumerate(data):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    result = self._recursive_mcid_search(item, new_path)
                    if result:
                        return result
            
            return ""
            
        except Exception as e:
            print(f"DEBUG: Error in recursive search at path '{path}': {e}")
            return ""
    
    def _generate_patient_id_hash(self, mcid: str) -> str:
        """Generate a secure hash from MCID for filename"""
        if not mcid:
            raise EpisodicMemoryError("Cannot generate hash from empty MCID")
        
        # Create SHA256 hash
        hash_object = hashlib.sha256(mcid.encode('utf-8'))
        hash_result = hash_object.hexdigest()[:16]  # Use first 16 characters
        print(f"DEBUG: Generated hash for MCID '{mcid}': {hash_result}")
        return hash_result
    
    def _get_memory_file_path(self, patient_hash: str) -> Path:
        """Get the file path for patient's episodic memory"""
        filename = f"{self.config.file_prefix}_{patient_hash}.json"
        file_path = self.storage_path / filename
        print(f"DEBUG: Memory file path: {file_path}")
        return file_path
    
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
                print(f"DEBUG: Memory file does not exist: {file_path}")
                return None
            
            with self._file_lock(file_path, 'r') as f:
                if f is None:
                    return None
                
                data = json.load(f)
                print(f"DEBUG: Loaded existing memory data: {type(data)}")
                
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
                print(f"DEBUG: No existing file to backup: {file_path}")
                return True
            
            backup_path = self._get_backup_file_path(patient_hash)
            shutil.copy2(file_path, backup_path)
            
            print(f"DEBUG: Created backup: {backup_path}")
            logger.debug(f"Created backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def _save_memory(self, file_path: Path, memory_data: Dict[str, Any], patient_hash: str) -> bool:
        """Save episodic memory data with backup"""
        try:
            print(f"DEBUG: Attempting to save memory to: {file_path}")
            print(f"DEBUG: Memory data type: {type(memory_data)}")
            print(f"DEBUG: Memory data preview: {json.dumps(memory_data, indent=2, default=str)[:500]}...")
            
            # Create backup of existing file
            backup_success = self._create_backup(file_path, patient_hash)
            print(f"DEBUG: Backup creation success: {backup_success}")
            
            # Write new data
            with self._file_lock(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Verify file was created
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"DEBUG: File saved successfully. Size: {file_size} bytes")
                logger.debug(f"Successfully saved episodic memory: {file_path}")
                return True
            else:
                print(f"DEBUG: File was not created: {file_path}")
                return False
            
        except Exception as e:
            print(f"DEBUG: Error saving memory file: {e}")
            logger.error(f"Error saving memory file {file_path}: {e}")
            return False
    
    def save_episodic_memory(self, deidentified_mcid: Dict[str, Any], 
                           entity_extraction: Dict[str, Any],
                           additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fixed save episodic memory with improved mcidList extraction
        
        Args:
            deidentified_mcid: Deidentified MCID data (may be wrapped in mcid_claims_data)
            entity_extraction: Extracted health entities
            additional_metadata: Optional additional metadata (ignored - for compatibility)
        
        Returns:
            Dict containing operation result and memory info
        """
        try:
            print("DEBUG: === STARTING EPISODIC MEMORY SAVE ===")
            print(f"DEBUG: Input deidentified_mcid keys: {list(deidentified_mcid.keys()) if isinstance(deidentified_mcid, dict) else 'Not a dict'}")
            print(f"DEBUG: Input entity_extraction: {entity_extraction}")
            
            # Fixed MCID extraction with multiple fallback methods
            mcid = self._extract_mcid_list(deidentified_mcid)
            print(f"DEBUG: Extracted MCID: '{mcid}'")
            
            if not mcid:
                error_msg = "No 'mcidList' found in deidentified data"
                print(f"DEBUG: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "mcid": "",
                    "operation": "none",
                    "debug_data": {
                        "input_keys": list(deidentified_mcid.keys()) if isinstance(deidentified_mcid, dict) else "not_dict",
                        "extraction_attempted": True
                    }
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
            print(f"DEBUG: Simplified entities: {simplified_entities}")
            
            # Create new visit entry with exact structure requested
            current_timestamp = datetime.now().isoformat()
            new_visit = {
                "id_type": "mcid",
                "mcid": mcid,
                "entity_extraction": simplified_entities,
                "timestamp": current_timestamp
            }
            print(f"DEBUG: New visit data: {new_visit}")
            
            # Check if file exists for returning patient
            existing_memory = self._load_existing_memory(file_path)
            print(f"DEBUG: Existing memory found: {existing_memory is not None}")
            
            if existing_memory:
                # Returning patient - append to existing data
                operation = "updated"
                print("DEBUG: Patient is returning - appending to existing data")
                
                # If existing data is not a list, convert it
                if isinstance(existing_memory, dict) and "id_type" in existing_memory:
                    # Old single visit format, convert to list
                    memory_data = [existing_memory, new_visit]
                    print("DEBUG: Converted single visit to list format")
                elif isinstance(existing_memory, list):
                    # Already a list of visits, append new visit
                    memory_data = existing_memory + [new_visit]
                    print(f"DEBUG: Appended to existing list. Total visits: {len(memory_data)}")
                else:
                    # Unknown format, start fresh
                    memory_data = [new_visit]
                    print("DEBUG: Unknown format found, starting fresh with list")
            else:
                # New patient - single visit
                operation = "created"
                memory_data = new_visit
                print("DEBUG: New patient - creating single visit entry")
            
            # Save memory data
            print("DEBUG: Attempting to save memory data...")
            save_success = self._save_memory(file_path, memory_data, patient_hash)
            print(f"DEBUG: Save operation success: {save_success}")
            
            # Calculate visit count
            visit_count = len(memory_data) if isinstance(memory_data, list) else 1
            
            result = {
                "success": save_success,
                "operation": operation,
                "patient_hash": patient_hash,
                "file_path": str(file_path),
                "mcid": mcid,
                "visit_count": visit_count,
                "timestamp": current_timestamp,
                "debug_info": {
                    "mcid_extracted": mcid,
                    "file_exists_after_save": file_path.exists() if save_success else False,
                    "memory_data_type": type(memory_data).__name__,
                    "simplified_entities": simplified_entities
                }
            }
            
            print(f"DEBUG: Final result: {result}")
            print("DEBUG: === EPISODIC MEMORY SAVE COMPLETED ===")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in save_episodic_memory: {str(e)}"
            print(f"DEBUG: Exception occurred: {error_msg}")
            logger.error(error_msg)
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
                "operation": "error",
                "debug_info": {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e)
                }
            }
    
    def load_episodic_memory(self, mcid: str) -> Optional[Dict[str, Any]]:
        """Load episodic memory for given MCID"""
        try:
            if not mcid:
                return None
            
            patient_hash = self._generate_patient_id_hash(mcid)
            file_path = self._get_memory_file_path(patient_hash)
            
            print(f"DEBUG: Loading memory from: {file_path}")
            memory_data = self._load_existing_memory(file_path)
            print(f"DEBUG: Loaded memory data: {memory_data is not None}")
            
            return memory_data
            
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

# Test function to verify the fix
def test_mcid_extraction():
    """Test function to verify mcidList extraction works correctly"""
    
    # Test with original API structure
    original_structure = {
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
    
    # Test with deidentified structure (wrapped in mcid_claims_data)
    deidentified_structure = {
        "mcid_claims_data": {
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
        "original_structure_preserved": True,
        "deidentification_timestamp": "2025-08-29T10:00:00",
        "data_type": "stable_mcid_claims",
        "processing_method": "stable"
    }
    
    test_entities = {
        "diabetics": "yes",
        "blood_pressure": "managed",
        "age": 45,
        "smoking": "no",
        "alcohol": "unknown"
    }
    
    config = EpisodicMemoryConfig(
        storage_directory="./test_episodic_memory",
        backup_directory="./test_episodic_memory_backup"
    )
    
    manager = EpisodicMemoryManager(config)
    
    print("=== TESTING ORIGINAL STRUCTURE ===")
    result1 = manager.save_episodic_memory(original_structure, test_entities)
    print(f"Original structure result: {result1}")
    
    print("\n=== TESTING DEIDENTIFIED STRUCTURE ===")
    result2 = manager.save_episodic_memory(deidentified_structure, test_entities)
    print(f"Deidentified structure result: {result2}")
    
    # Test loading
    if result1["success"]:
        print(f"\n=== TESTING LOAD FOR MCID: {result1['mcid']} ===")
        loaded_memory = manager.load_episodic_memory(result1['mcid'])
        print(f"Loaded memory: {json.dumps(loaded_memory, indent=2) if loaded_memory else 'None'}")
    
    return result1, result2

if __name__ == "__main__":
    # Run the test
    test_result1, test_result2 = test_mcid_extraction()
    
    # Print summary
    print(f"\n=== TEST SUMMARY ===")
    print(f"Original structure test: {'PASSED' if test_result1['success'] else 'FAILED'}")
    print(f"Deidentified structure test: {'PASSED' if test_result2['success'] else 'FAILED'}")
    
    if test_result1['success'] or test_result2['success']:
        config = EpisodicMemoryConfig(storage_directory="./test_episodic_memory")
        manager = EpisodicMemoryManager(config)
        stats = manager.get_statistics()
        print(f"Memory statistics: {stats}")
