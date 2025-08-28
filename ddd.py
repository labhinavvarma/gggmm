"""
CRITICAL SECURITY WARNING:
This implementation stores healthcare data in local files without encryption.
This approach has serious HIPAA compliance risks and should NOT be used in production.
Consider encrypted databases or healthcare-compliant storage solutions instead.
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
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
    max_history_entries: int = 50
    encrypt_files: bool = False  # Set to False since not implemented
    backup_directory: str = "./episodic_memory_backup"
    compression_enabled: bool = True
    retention_days: int = 365

class EpisodicMemoryError(Exception):
    """Custom exception for episodic memory operations"""
    pass

class EpisodicMemoryManager:
    """
    Manages episodic memory for patient healthcare data
    
    WARNING: This implementation stores PHI in local files without encryption.
    This violates HIPAA requirements and should not be used in production.
    """
    
    def __init__(self, config: EpisodicMemoryConfig):
        self.config = config
        self.storage_path = Path(config.storage_directory)
        self.backup_path = Path(config.backup_directory)
        
        # Ensure directories exist
        self._setup_directories()
        
        logger.warning("SECURITY WARNING: Episodic memory stores PHI in unencrypted files")
    
    def _setup_directories(self):
        """Setup storage and backup directories with secure permissions"""
        try:
            self.storage_path.mkdir(exist_ok=True, parents=True)
            self.backup_path.mkdir(exist_ok=True, parents=True)
            
            # Set secure permissions (Unix systems only)
            try:
                os.chmod(self.storage_path, 0o700)  # Owner read/write/execute only
                os.chmod(self.backup_path, 0o700)
            except OSError as e:
                logger.warning(f"Could not set directory permissions: {e}")
                
        except Exception as e:
            raise EpisodicMemoryError(f"Failed to setup directories: {e}")
    
    def _extract_mcid_list(self, deidentified_mcid: Dict[str, Any]) -> List[str]:
        """Extract MCID list from deidentified MCID JSON structure"""
        mcid_list = []
        
        try:
            # Navigate the MCID data structure
            mcid_data = deidentified_mcid.get("mcid_claims_data", {})
            
            if not mcid_data:
                logger.warning("No MCID claims data found in deidentified_mcid")
                return mcid_list
            
            # Recursive function to extract MCIDs
            def extract_mcids_recursive(data: Any, path: str = ""):
                if isinstance(data, dict):
                    # Look for MCID fields (case-insensitive)
                    for key, value in data.items():
                        key_lower = key.lower()
                        if any(field in key_lower for field in ['mcid', 'member_id', 'patient_id', 'mbr_id', 'id']):
                            if value and str(value).strip():
                                mcid_value = str(value).strip()
                                if mcid_value not in mcid_list and mcid_value != "[MASKED_NAME]":
                                    mcid_list.append(mcid_value)
                                    logger.debug(f"Found MCID: {mcid_value} at {path}.{key}")
                        
                        # Continue recursive search
                        new_path = f"{path}.{key}" if path else key
                        extract_mcids_recursive(value, new_path)
                
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        new_path = f"{path}[{i}]" if path else f"[{i}]"
                        extract_mcids_recursive(item, new_path)
            
            extract_mcids_recursive(mcid_data)
            
            logger.info(f"Extracted {len(mcid_list)} unique MCIDs")
            return mcid_list
            
        except Exception as e:
            logger.error(f"Error extracting MCID list: {e}")
            return []
    
    def _generate_patient_id_hash(self, mcid_list: List[str]) -> str:
        """Generate a secure hash from MCID list for filename"""
        if not mcid_list:
            raise EpisodicMemoryError("Cannot generate hash from empty MCID list")
        
        # Sort MCIDs for consistent hashing
        sorted_mcids = sorted(set(mcid_list))
        combined_mcids = "|".join(sorted_mcids)
        
        # Create SHA256 hash
        hash_object = hashlib.sha256(combined_mcids.encode('utf-8'))
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
        """Load existing episodic memory file with error handling"""
        try:
            if not file_path.exists():
                return None
            
            with self._file_lock(file_path, 'r') as f:
                if f is None:
                    return None
                
                data = json.load(f)
                
                # Validate required fields
                if not isinstance(data, dict):
                    logger.error(f"Invalid memory file format: {file_path}")
                    return None
                
                # Ensure required fields exist
                required_fields = {
                    'mcid_list': [],
                    'entity_extraction': {},
                    'history': [],
                    'metadata': {}
                }
                
                for field, default in required_fields.items():
                    if field not in data:
                        logger.warning(f"Missing required field '{field}' in memory file: {file_path}")
                        data[field] = default
                
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
        """Save episodic memory data with error handling and backup"""
        try:
            # Create backup of existing file
            backup_success = self._create_backup(file_path, patient_hash)
            if not backup_success:
                logger.warning("Backup creation failed, proceeding with save")
            
            # Write new data with file locking
            with self._file_lock(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Set secure file permissions
            try:
                os.chmod(file_path, 0o600)  # Owner read/write only
            except OSError as e:
                logger.warning(f"Could not set file permissions: {e}")
            
            logger.debug(f"Successfully saved episodic memory: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory file {file_path}: {e}")
            return False
    
    def save_episodic_memory(self, deidentified_mcid: Dict[str, Any], 
                           entity_extraction: Dict[str, Any],
                           additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save or update episodic memory for a patient
        
        Args:
            deidentified_mcid: Deidentified MCID data
            entity_extraction: Extracted health entities
            additional_metadata: Optional additional metadata to store
        
        Returns:
            Dict containing operation result and memory info
        """
        try:
            # Extract MCID list
            mcid_list = self._extract_mcid_list(deidentified_mcid)
            
            if not mcid_list:
                return {
                    "success": False,
                    "error": "No MCIDs found in deidentified data",
                    "mcid_list": [],
                    "operation": "none"
                }
            
            # Generate patient hash for filename
            patient_hash = self._generate_patient_id_hash(mcid_list)
            file_path = self._get_memory_file_path(patient_hash)
            
            # Load existing memory
            existing_memory = self._load_existing_memory(file_path)
            current_timestamp = datetime.now().isoformat()
            
            if existing_memory:
                # Update existing memory
                operation = "updated"
                
                # Merge MCID lists (keep unique values)
                existing_mcids = set(existing_memory.get('mcid_list', []))
                new_mcids = set(mcid_list)
                merged_mcids = sorted(list(existing_mcids.union(new_mcids)))
                
                # Create history entry
                history_entry = {
                    "timestamp": current_timestamp,
                    "entity_extraction": entity_extraction,
                    "mcid_list": mcid_list,
                    "metadata": additional_metadata or {}
                }
                
                # Maintain history limit
                history = existing_memory.get('history', [])
                history.append(history_entry)
                if len(history) > self.config.max_history_entries:
                    history = history[-self.config.max_history_entries:]
                
                # Update memory structure
                memory_data = {
                    "patient_hash": patient_hash,
                    "mcid_list": merged_mcids,
                    "entity_extraction": entity_extraction,  # Always use latest
                    "first_seen": existing_memory.get('first_seen', current_timestamp),
                    "last_updated": current_timestamp,
                    "update_count": existing_memory.get('update_count', 0) + 1,
                    "history": history,
                    "metadata": {
                        **existing_memory.get('metadata', {}),
                        **(additional_metadata or {})
                    },
                    "version": "1.0"
                }
                
            else:
                # Create new memory
                operation = "created"
                
                memory_data = {
                    "patient_hash": patient_hash,
                    "mcid_list": mcid_list,
                    "entity_extraction": entity_extraction,
                    "first_seen": current_timestamp,
                    "last_updated": current_timestamp,
                    "update_count": 1,
                    "history": [
                        {
                            "timestamp": current_timestamp,
                            "entity_extraction": entity_extraction,
                            "mcid_list": mcid_list,
                            "metadata": additional_metadata or {}
                        }
                    ],
                    "metadata": additional_metadata or {},
                    "version": "1.0"
                }
            
            # Save memory data
            save_success = self._save_memory(file_path, memory_data, patient_hash)
            
            return {
                "success": save_success,
                "operation": operation,
                "patient_hash": patient_hash,
                "file_path": str(file_path),
                "mcid_list": mcid_list,
                "update_count": memory_data["update_count"],
                "first_seen": memory_data["first_seen"],
                "last_updated": memory_data["last_updated"],
                "history_entries": len(memory_data["history"])
            }
            
        except Exception as e:
            logger.error(f"Error in save_episodic_memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": "error"
            }
    
    def load_episodic_memory(self, mcid_list: List[str]) -> Optional[Dict[str, Any]]:
        """Load episodic memory for given MCID list"""
        try:
            if not mcid_list:
                return None
            
            patient_hash = self._generate_patient_id_hash(mcid_list)
            file_path = self._get_memory_file_path(patient_hash)
            
            return self._load_existing_memory(file_path)
            
        except Exception as e:
            logger.error(f"Error loading episodic memory: {e}")
            return None
    
    def load_episodic_memory_by_hash(self, patient_hash: str) -> Optional[Dict[str, Any]]:
        """Load episodic memory by patient hash"""
        try:
            file_path = self._get_memory_file_path(patient_hash)
            return self._load_existing_memory(file_path)
        except Exception as e:
            logger.error(f"Error loading episodic memory by hash: {e}")
            return None
    
    def get_patient_history(self, mcid_list: List[str], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get patient's historical entries"""
        memory = self.load_episodic_memory(mcid_list)
        if not memory:
            return []
        
        history = memory.get('history', [])
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_patient_summary(self, mcid_list: List[str]) -> Optional[Dict[str, Any]]:
        """Get a summary of patient's episodic memory"""
        memory = self.load_episodic_memory(mcid_list)
        if not memory:
            return None
        
        history = memory.get('history', [])
        entity_extraction = memory.get('entity_extraction', {})
        
        return {
            "patient_hash": memory.get('patient_hash'),
            "first_seen": memory.get('first_seen'),
            "last_updated": memory.get('last_updated'),
            "total_visits": len(history),
            "update_count": memory.get('update_count', 0),
            "current_conditions": entity_extraction.get('medical_conditions', []),
            "current_medications": entity_extraction.get('medications_identified', []),
            "diabetes_status": entity_extraction.get('diabetics', 'unknown'),
            "bp_status": entity_extraction.get('blood_pressure', 'unknown'),
            "smoking_status": entity_extraction.get('smoking', 'unknown')
        }
    
    def delete_patient_memory(self, mcid_list: List[str]) -> bool:
        """Delete patient's episodic memory (for compliance purposes)"""
        try:
            if not mcid_list:
                return False
            
            patient_hash = self._generate_patient_id_hash(mcid_list)
            file_path = self._get_memory_file_path(patient_hash)
            
            if file_path.exists():
                # Create final backup before deletion
                self._create_backup(file_path, patient_hash)
                
                # Delete the file
                file_path.unlink()
                logger.info(f"Deleted episodic memory for patient hash: {patient_hash}")
                return True
            else:
                logger.warning(f"No memory file found for patient hash: {patient_hash}")
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
                        last_updated_str = memory.get('last_updated', '')
                        if last_updated_str:
                            last_updated = datetime.fromisoformat(last_updated_str)
                            if last_updated < cutoff_date:
                                # Create backup before deletion
                                patient_hash = memory.get('patient_hash', 'unknown')
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
        """List all patients in episodic memory (for admin purposes)"""
        patients = []
        
        try:
            for file_path in self.storage_path.glob(f"{self.config.file_prefix}_*.json"):
                try:
                    memory = self._load_existing_memory(file_path)
                    if memory:
                        patients.append({
                            "patient_hash": memory.get('patient_hash'),
                            "first_seen": memory.get('first_seen'),
                            "last_updated": memory.get('last_updated'),
                            "update_count": memory.get('update_count', 0),
                            "history_entries": len(memory.get('history', [])),
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
                    "storage_directory": str(self.storage_path),
                    "backup_directory": str(self.backup_path)
                }
            
            total_updates = sum(p.get('update_count', 0) for p in patients)
            total_history_entries = sum(p.get('history_entries', 0) for p in patients)
            
            # Get oldest and newest
            oldest_patient = min(patients, key=lambda x: x.get('first_seen', ''))
            newest_patient = max(patients, key=lambda x: x.get('last_updated', ''))
            
            return {
                "total_patients": len(patients),
                "total_memory_files": len(list(self.storage_path.glob(f"{self.config.file_prefix}_*.json"))),
                "total_updates": total_updates,
                "total_history_entries": total_history_entries,
                "oldest_patient_first_seen": oldest_patient.get('first_seen'),
                "newest_patient_updated": newest_patient.get('last_updated'),
                "storage_directory": str(self.storage_path),
                "backup_directory": str(self.backup_path),
                "retention_days": self.config.retention_days
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
