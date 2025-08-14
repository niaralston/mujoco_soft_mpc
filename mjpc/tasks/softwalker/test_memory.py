#!/usr/bin/env python3
"""
Test script to check MuJoCo memory allocation and usage
"""

import mujoco
import sys
import os
import time

def test_model_memory(xml_path, description):
    """Test a model and report memory usage"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"XML file: {xml_path}")
    print(f"{'='*60}")
    
    try:
        # Load the model
        print(f"Loading model...")
        model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"  ✅ Model loaded successfully!")
        
        # Create data
        print(f"Creating simulation data...")
        data = mujoco.MjData(model)
        print(f"  ✅ Data created successfully!")
        
        # Print model size parameters
        print(f"\nModel size parameters:")
        print(f"  nstack: {getattr(model, 'nstack', 'N/A')}")
        print(f"  memory: {getattr(model, 'memory', 'N/A')}")
        print(f"  njmax: {getattr(model, 'njmax', 'N/A')}")
        print(f"  nconmax: {getattr(model, 'nconmax', 'N/A')}")
        
        # Check if flexible components exist
        if hasattr(model, 'nflex') and model.nflex > 0:
            print(f"  nflex: {model.nflex}")
            print(f"  ✅ Model has flexible components!")
        else:
            print(f"  ❌ No flexible components found")
        
        # Try to step the simulation
        print(f"\nTesting simulation step...")
        try:
            mujoco.mj_step(model, data)
            print(f"  ✅ Simulation step successful!")
        except Exception as e:
            print(f"  ❌ Simulation step failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False

def main():
    print("MuJoCo Memory Testing Tool")
    print(f"MuJoCo version: {mujoco.__version__}")
    
    # Test different configurations
    test_configs = [
        ("softwalker_modified.xml", "Current config with memory parameter"),
        ("softwalker_fixed.xml", "Fixed version (no flexible components)"),
        ("inflatable_walker.xml", "Inflatable walker (different memory config)")
    ]
    
    for xml_file, description in test_configs:
        xml_path = os.path.join(os.path.dirname(__file__), xml_file)
        if os.path.exists(xml_path):
            test_model_memory(xml_path, description)
        else:
            print(f"\n❌ File not found: {xml_path}")
    
    print(f"\n{'='*60}")
    print("Memory testing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 