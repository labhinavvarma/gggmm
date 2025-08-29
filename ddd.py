"""
Unit Tests for Episodic Memory Manager

SECURITY WARNING: These tests use mock healthcare data.
Never use real PHI in test environments.
"""

import unittest
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from episodic_memory_manager import EpisodicMemoryManager, EpisodicMemoryConfig, EpisodicMemoryError

class TestEpisodicMemoryManager(unittest.TestCase):
    """Test cases for EpisodicMemoryManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.backup_dir = tempfile.mkdtemp()
        
        self.config = EpisodicMemoryConfig(
            storage_directory=self.test_dir,
            backup_directory=self.backup_dir,
            max_history_entries=5,  # Small for testing
            retention_days=30
        )
        
        self.memory_manager = EpisodicMemoryManager(self.config)
        
        # Mock deidentified MCID data
        self.mock_deidentified_mcid = {
            "mcid_claims_data": {
                "patient_records": [
                    {"mcid": "TEST12345", "member_id": "MBR001"},
                    {"patient_id": "PAT001", "mbr_id": "MBR001"}
                ]
            }
        }
        
        # Mock entity extraction
        self.mock_entity_extraction = {
            "diabetics": "yes",
            "age_group": "adult",
            "age": 45,
            "smoking": "no",
            "blood_pressure": "managed",
            "medical_conditions": ["Diabetes Type 2", "Hypertension"],
            "medications_identified": [
                {"ndc": "12345-678-90", "label_name": "Metformin"},
                {"ndc": "09876-543-21", "label_name": "Lisinopril"}
            ]
        }
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.backup_dir)
    
    def test_extract_mcid_list(self):
        """Test MCID extraction from deidentified data"""
        mcid_list = self.memory_manager._extract_mcid_list(self.mock_deidentified_mcid)
        
        expected_mcids = ["TEST12345", "MBR001", "PAT001"]
        self.assertEqual(len(mcid_list), 3)
        for mcid in expected_mcids:
            self.assertIn(mcid, mcid_list)
    
    def test_generate_patient_hash(self):
        """Test patient hash generation"""
        mcid_list = ["TEST12345", "MBR001", "PAT001"]
        hash1 = self.memory_manager._generate_patient_id_hash(mcid_list)
        hash2 = self.memory_manager._generate_patient_id_hash(mcid_list)
        
        # Hash should be consistent
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)  # SHA256 truncated to 16 chars
        
        # Different MCID lists should produce different hashes
        different_list = ["DIFFERENT123", "MCID456"]
        hash3 = self.memory_manager._generate_patient_id_hash(different_list)
        self.assertNotEqual(hash1, hash3)
    
    def test_save_new_episodic_memory(self):
        """Test saving new episodic memory"""
        result = self.memory_manager.save_episodic_memory(
            deidentified_mcid=self.mock_deidentified_mcid,
            entity_extraction=self.mock_entity_extraction,
            additional_metadata={"test": "data"}
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["operation"], "created")
        self.assertEqual(result["update_count"], 1)
        self.assertIn("patient_hash", result)
        self.assertIn("file_path", result)
        
        # Verify file was created
        file_path = Path(result["file_path"])
        self.assertTrue(file_path.exists())
    
    def test_update_existing_episodic_memory(self):
        """Test updating existing episodic memory"""
        # First save
        result1 = self.memory_manager.save_episodic_memory(
            deidentified_mcid=self.mock_deidentified_mcid,
            entity_extraction=self.mock_entity_extraction
        )
        self.assertTrue(result1["success"])
        self.assertEqual(result1["operation"], "created")
        
        # Update with new entity data
        updated_entities = {
            **self.mock_entity_extraction,
            "blood_pressure": "diagnosed",  # Changed
            "medical_conditions": ["Diabetes Type 2", "Hypertension", "High Cholesterol"]  # Added condition
        }
        
        result2 = self.memory_manager.save_episodic_memory(
            deidentified_mcid=self.mock_deidentified_mcid,
            entity_extraction=updated_entities
        )
        
        self.assertTrue(result2["success"])
        self.assertEqual(result2["operation"], "updated")
        self.assertEqual(result2["update_count"], 2)
        self.assertEqual(result2["patient_hash"], result1["patient_hash"])  # Same patient
        self.assertEqual(result2["history_entries"], 2)
    
    def test_load_episodic_memory(self):
        """Test loading episodic memory"""
        # Save first
        save_result = self.memory_manager.save_episodic_memory(
            deidentified_mcid=self.mock_deidentified_mcid,
            entity_extraction=self.mock_entity_extraction
        )
        
        mcid_list = save_result["mcid_list"]
        
        # Load
        loaded_memory = self.memory_manager.load_episodic_memory(mcid_list)
        
        self.assertIsNotNone(loaded_memory)
        self.assertEqual(loaded_memory["patient_hash"], save_result["patient_hash"])
        self.assertEqual(loaded_memory["mcid_list"], mcid_list)
        self.assertEqual(loaded_memory["entity_extraction"], self.mock_entity_extraction)
        self.assertIn("history", loaded_memory)
        self.assertEqual(len(loaded_memory["history"]), 1)
    
    def test_get_patient_history(self):
        """Test retrieving patient history"""
        mcid_list = ["TEST12345", "MBR001"]
        
        # Create multiple visits
        for i in range(3):
            entities = {
                **self.mock_entity_extraction,
                "visit_number": i + 1
            }
            self.memory_manager.save_episodic_memory(
                deidentified_mcid=self.mock_deidentified_mcid,
                entity_extraction=entities
            )
        
        # Get history
        history = self.memory_manager.get_patient_history(mcid_list)
        self.assertEqual(len(history), 3)
        
        # Test with limit
        limited_history = self.memory_manager.get_patient_history(mcid_list, limit=2)
        self.assertEqual(len(limited_history), 2)
    
    def test_get_patient_summary(self):
        """Test patient summary generation"""
        # Save memory
        save_result = self.memory_manager.save_episodic_memory(
            deidentified_mcid=self.mock_deidentified_mcid,
            entity_extraction=self.mock_entity_extraction
        )
        
        mcid_list = save_result["mcid_list"]
        summary = self.memory_manager.get_patient_summary(mcid_list)
        
        self.assertIsNotNone(summary)
        self.assertEqual(summary["patient_hash"], save_result["patient_hash"])
        self.assertEqual(summary["total_visits"], 1)
        self.assertEqual(summary["diabetes_status"], "yes")
        self.assertEqual(summary["bp_status"], "managed")
        self.assertIn("current_conditions", summary)
        self.assertIn("current_medications", summary)
    
    def test_delete_patient_memory(self):
        """Test patient memory deletion"""
        # Save memory
        save_result = self.memory_manager.save_episodic_memory(
            deidentified_mcid=self.mock_deidentified_mcid,
            entity_extraction=self.mock_entity_extraction
        )
        
        mcid_list = save_result["mcid_list"]
        file_path = Path(save_result["file_path"])
        
        # Verify file exists
        self.assertTrue(file_path.exists())
        
        # Delete memory
        delete_success = self.memory_manager.delete_patient_memory(mcid_list)
        self.assertTrue(delete_success)
        
        # Verify file deleted
        self.assertFalse(file_path.exists())
        
        # Verify cannot load
        loaded_memory = self.memory_manager.load_episodic_memory(mcid_list)
        self.assertIsNone(loaded_memory)
    
    def test_history_limit(self):
        """Test history entry limit enforcement"""
        mcid_list = ["TEST12345"]
        
        # Create more visits than the limit (5)
        for i in range(8):
            entities = {
                **self.mock_entity_extraction,
                "visit_number": i + 1
            }
            self.memory_manager.save_episodic_memory(
                deidentified_mcid=self.mock_deidentified_mcid,
                entity_extraction=entities
            )
        
        # Load and verify history was limited
        loaded_memory = self.memory_manager.load_episodic_memory(mcid_list)
        self.assertIsNotNone(loaded_memory)
        self.assertEqual(len(loaded_memory["history"]), 5)  # Limited to max_history_entries
        self.assertEqual(loaded_memory["update_count"], 8)  # But update count is preserved
        
        # Verify we kept the most recent entries
        last_entry = loaded_memory["history"][-1]
        self.assertEqual(last_entry["entity_extraction"]["visit_number"], 8)
    
    def test_cleanup_old_memories(self):
        """Test cleanup of old memory files"""
        # Create a memory file
        save_result = self.memory_manager.save_episodic_memory(
            deidentified_mcid=self.mock_deidentified_mcid,
            entity_extraction=self.mock_entity_extraction
        )
        
        file_path = Path(save_result["file_path"])
        self.assertTrue(file_path.exists())
        
        # Test cleanup with 0 days (should remove file)
        cleanup_result = self.memory_manager.cleanup_old_memories(days_old=0)
        
        self.assertEqual(cleanup_result["removed_count"], 1)
        self.assertEqual(cleanup_result["error_count"], 0)
        self.assertFalse(file_path.exists())
    
    def test_list_all_patients(self):
        """Test listing all patients"""
        # Create multiple patients
        mcid_data_1 = {
            "mcid_claims_data": {
                "patient_records": [{"mcid": "PATIENT1", "member_id": "MBR001"}]
            }
        }
        
        mcid_data_2 = {
            "mcid_claims_data": {
                "patient_records": [{"mcid": "PATIENT2", "member_id": "MBR002"}]
            }
        }
        
        self.memory_manager.save_episodic_memory(
            deidentified_mcid=mcid_data_1,
            entity_extraction=self.mock_entity_extraction
        )
        
        self.memory_manager.save_episodic_memory(
            deidentified_mcid=mcid_data_2,
            entity_extraction=self.mock_entity_extraction
        )
        
        # List patients
        patients = self.memory_manager.list_all_patients()
        self.assertEqual(len(patients), 2)
        
        for patient in patients:
            self.assertIn("patient_hash", patient)
            self.assertIn("first_seen", patient)
            self.assertIn("last_updated", patient)
            self.assertIn("update_count", patient)
    
    def test_statistics(self):
        """Test getting episodic memory statistics"""
        # Create some memory data
        for i in range(3):
            mcid_data = {
                "mcid_claims_data": {
                    "patient_records": [{"mcid": f"PATIENT{i}", "member_id": f"MBR{i:03d}"}]
                }
            }
            self.memory_manager.save_episodic_memory(
                deidentified_mcid=mcid_data,
                entity_extraction=self.mock_entity_extraction
            )
        
        # Get statistics
        stats = self.memory_manager.get_statistics()
        
        self.assertEqual(stats["total_patients"], 3)
        self.assertEqual(stats["total_memory_files"], 3)
        self.assertIn("storage_directory", stats)
        self.assertIn("backup_directory", stats)
        self.assertIn("oldest_patient_first_seen", stats)
        self.assertIn("newest_patient_updated", stats)
    
    def test_invalid_mcid_data(self):
        """Test handling of invalid MCID data"""
        # Empty MCID data
        result = self.memory_manager.save_episodic_memory(
            deidentified_mcid={},
            entity_extraction=self.mock_entity_extraction
        )
        
        self.assertFalse(result["success"])
        self.assertEqual(result["operation"], "none")
        self.assertIn("No MCIDs found", result["error"])
        
        # MCID data without claims
        result = self.memory_manager.save_episodic_memory(
            deidentified_mcid={"mcid_claims_data": {}},
            entity_extraction=self.mock_entity_extraction
        )
        
        self.assertFalse(result["success"])
        self.assertEqual(result["operation"], "none")
    
    def test_file_corruption_handling(self):
        """Test handling of corrupted memory files"""
        # Save normal memory
        save_result = self.memory_manager.save_episodic_memory(
            deidentified_mcid=self.mock_deidentified_mcid,
            entity_extraction=self.mock_entity_extraction
        )
        
        file_path = Path(save_result["file_path"])
        
        # Corrupt the file
        with open(file_path, 'w') as f:
            f.write("corrupted json data {")
        
        # Try to load
        mcid_list = save_result["mcid_list"]
        loaded_memory = self.memory_manager.load_episodic_memory(mcid_list)
        
        # Should return None for corrupted file
        self.assertIsNone(loaded_memory)

class TestEpisodicMemoryConfig(unittest.TestCase):
    """Test cases for EpisodicMemoryConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EpisodicMemoryConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.storage_type, "file")
        self.assertEqual(config.max_history_entries, 50)
        self.assertFalse(config.encryption_enabled)  # Default to False for development
        self.assertTrue(config.compression_enabled)

class TestEpisodicMemoryIntegration(unittest.TestCase):
    """Integration tests for episodic memory system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.backup_dir = tempfile.mkdtemp()
        
        self.config = EpisodicMemoryConfig(
            storage_directory=self.test_dir,
            backup_directory=self.backup_dir,
            max_history_entries=10,
            retention_days=30
        )
        
        self.memory_manager = EpisodicMemoryManager(self.config)
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.backup_dir)
    
    def test_patient_journey_simulation(self):
        """Simulate a complete patient journey over multiple visits"""
        
        # Patient's MCID data (consistent across visits)
        mcid_data = {
            "mcid_claims_data": {
                "patient_records": [
                    {"mcid": "JOURNEY123", "member_id": "MBR999"}
                ]
            }
        }
        
        # Visit 1: Initial diabetes diagnosis
        visit1_entities = {
            "diabetics": "yes",
            "age": 50,
            "age_group": "middle_aged",
            "smoking": "no",
            "blood_pressure": "unknown",
            "medical_conditions": ["Type 2 Diabetes"],
            "medications_identified": [
                {"ndc": "12345-678-90", "label_name": "Metformin"}
            ]
        }
        
        result1 = self.memory_manager.save_episodic_memory(
            deidentified_mcid=mcid_data,
            entity_extraction=visit1_entities,
            additional_metadata={"visit_type": "initial_diagnosis"}
        )
        
        self.assertTrue(result1["success"])
        self.assertEqual(result1["operation"], "created")
        
        # Visit 2: Add hypertension
        visit2_entities = {
            **visit1_entities,
            "blood_pressure": "diagnosed",
            "medical_conditions": ["Type 2 Diabetes", "Hypertension"],
            "medications_identified": [
                {"ndc": "12345-678-90", "label_name": "Metformin"},
                {"ndc": "09876-543-21", "label_name": "Lisinopril"}
            ]
        }
        
        result2 = self.memory_manager.save_episodic_memory(
            deidentified_mcid=mcid_data,
            entity_extraction=visit2_entities,
            additional_metadata={"visit_type": "follow_up"}
        )
        
        self.assertTrue(result2["success"])
        self.assertEqual(result2["operation"], "updated")
        self.assertEqual(result2["update_count"], 2)
        
        # Visit 3: Medication adjustment
        visit3_entities = {
            **visit2_entities,
            "blood_pressure": "managed",
            "medications_identified": [
                {"ndc": "12345-678-90", "label_name": "Metformin"},
                {"ndc": "09876-543-21", "label_name": "Lisinopril"},
                {"ndc": "11111-222-33", "label_name": "Atorvastatin"}  # Added statin
            ]
        }
        
        result3 = self.memory_manager.save_episodic_memory(
            deidentified_mcid=mcid_data,
            entity_extraction=visit3_entities,
            additional_metadata={"visit_type": "medication_adjustment"}
        )
        
        self.assertTrue(result3["success"])
        self.assertEqual(result3["operation"], "updated")
        self.assertEqual(result3["update_count"], 3)
        
        # Verify complete patient journey
        mcid_list = result3["mcid_list"]
        patient_memory = self.memory_manager.load_episodic_memory(mcid_list)
        
        self.assertIsNotNone(patient_memory)
        self.assertEqual(len(patient_memory["history"]), 3)
        
        # Check progression
        history = patient_memory["history"]
        
        # Visit 1: Only diabetes
        self.assertEqual(len(history[0]["entity_extraction"]["medical_conditions"]), 1)
        self.assertEqual(len(history[0]["entity_extraction"]["medications_identified"]), 1)
        
        # Visit 2: Diabetes + Hypertension
        self.assertEqual(len(history[1]["entity_extraction"]["medical_conditions"]), 2)
        self.assertEqual(len(history[1]["entity_extraction"]["medications_identified"]), 2)
        
        # Visit 3: Same conditions, more medications
        self.assertEqual(len(history[2]["entity_extraction"]["medical_conditions"]), 2)
        self.assertEqual(len(history[2]["entity_extraction"]["medications_identified"]), 3)
        
        # Verify metadata preserved
        self.assertEqual(history[0]["metadata"]["visit_type"], "initial_diagnosis")
        self.assertEqual(history[1]["metadata"]["visit_type"], "follow_up")
        self.assertEqual(history[2]["metadata"]["visit_type"], "medication_adjustment")
        
        # Test patient summary
        summary = self.memory_manager.get_patient_summary(mcid_list)
        self.assertEqual(summary["total_visits"], 3)
        self.assertEqual(summary["diabetes_status"], "yes")
        self.assertEqual(summary["bp_status"], "managed")

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
