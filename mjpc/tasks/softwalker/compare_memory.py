#!/usr/bin/env python3
"""
Compare different MuJoCo memory allocation strategies
"""

import mujoco
import sys
import os
import gc

def test_memory_config(xml_path, description):
    """Test a specific memory configuration"""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"XML file: {xml_path}")
    print(f"{'='*80}")
    
    try:
        # Load the model
        print(f"Loading model...")
        model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"  ✅ Model loaded successfully!")
        
        # Create data
        print(f"Creating simulation data...")
        data = mujoco.MjData(model)
        print(f"  ✅ Data created successfully!")
        
        # Print ALL size parameters
        print(f"\n📊 Model size parameters:")
        size_attrs = ['nstack', 'memory', 'njmax', 'nconmax', 'nuserdata', 'nkey', 
                     'nuser_body', 'nuser_jnt', 'nuser_geom', 'nuser_site', 
                     'nuser_cam', 'nuser_actuator', 'nuser_sensor', 'nuser_tendon', 
                     'nuser_contact', 'nuser_equality', 'nuser_blackbody', 
                     'nuser_text', 'nuser_mesh', 'nuser_hfield', 'nuser_skin', 'nuser_plugin']
        
        for attr in size_attrs:
            value = getattr(model, attr, 'N/A')
            if value != 'N/A' and value != -1:
                print(f"  {attr}: {value}")
        
        # Check flexible components
        if hasattr(model, 'nflex') and model.nflex > 0:
            print(f"  nflex: {model.nflex}")
            print(f"  ✅ Model has flexible components!")
        else:
            print(f"  ❌ No flexible components found")
        
        # Test simulation step
        print(f"\n🧪 Testing simulation step...")
        try:
            mujoco.mj_step(model, data)
            print(f"  ✅ Simulation step successful!")
            
            # Try multiple steps to see if memory holds up
            print(f"  🔄 Testing multiple steps...")
            for i in range(10):
                mujoco.mj_step(model, data)
            print(f"  ✅ 10 simulation steps successful!")
            
        except Exception as e:
            print(f"  ❌ Simulation step failed: {e}")
        
        # Clean up
        del model, data
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False

def create_test_xmls():
    """Create test XML files with different memory configurations"""
    
    # Test 1: High nstack approach
    test1_content = '''<mujoco model="Test1_HighNstack">
  <option solver="CG" tolerance="1e-6" timestep="1e-3" integrator="implicitfast"/>
  <size nstack="1000000000" njmax="2000" nconmax="500"/>
  <statistic extent="2" center="0 0 1"/>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
    <body name="test" pos="0 0 1">
      <geom name="sphere" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>'''
    
    # Test 2: Memory parameter approach (24GB = 24 * 1024^3 = 25769803776)
    test2_content = '''<mujoco model="Test2_MemoryParam">
  <option solver="CG" tolerance="1e-6" timestep="1e-3" integrator="implicitfast"/>
  <size memory="25769803776" njmax="2000" nconmax="500"/>
  <statistic extent="2" center="0 0 1"/>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
    <body name="test" pos="0 0 1">
      <geom name="sphere" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>'''
    
    # Test 3: Ultra high nstack approach
    test3_content = '''<mujoco model="Test3_UltraHighNstack">
  <option solver="CG" tolerance="1e-6" timestep="1e-3" integrator="implicitfast"/>
  <size nstack="10000000000" njmax="10000" nconmax="1000"/>
  <statistic extent="2" center="0 0 1"/>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
    <body name="test" pos="0 0 1">
      <geom name="sphere" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>'''
    
    # Write test files
    test_files = [
        ("test1_high_nstack.xml", test1_content, "High nstack approach (1B)"),
        ("test2_memory_param.xml", test2_content, "Memory parameter approach (24GB)"),
        ("test3_ultra_high_nstack.xml", test3_content, "Ultra high nstack approach (10B)")
    ]
    
    for filename, content, description in test_files:
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Created test file: {filename}")
    
    return test_files

def main():
    print("🧠 MuJoCo Memory Allocation Comparison Tool")
    print(f"MuJoCo version: {mujoco.__version__}")
    
    # Create test XML files
    print("\n📝 Creating test XML files...")
    test_files = create_test_xmls()
    
    # Test each configuration
    for filename, content, description in test_files:
        test_memory_config(filename, description)
    
    # Test your actual working files for comparison
    print(f"\n{'='*80}")
    print("COMPARING WITH WORKING MODELS")
    print(f"{'='*80}")
    
    working_models = [
        ("softwalker_fixed.xml", "Fixed version (no flexible components)"),
        ("inflatable_walker.xml", "Inflatable walker (working flexible model)")
    ]
    
    for xml_file, description in working_models:
        if os.path.exists(xml_file):
            test_memory_config(xml_file, description)
        else:
            print(f"\n❌ File not found: {xml_file}")
    
    print(f"\n{'='*80}")
    print("🎯 MEMORY COMPARISON SUMMARY")
    print(f"{'='*80}")
    print("This test shows you:")
    print("1. Which memory parameters actually get allocated")
    print("2. Whether nstack or memory parameter gives more capacity")
    print("3. How your working models compare")
    print("4. Which approach is best for your flexible components")
    print(f"{'='*80}")
    
    # Clean up test files
    print("\n🧹 Cleaning up test files...")
    for filename, content, description in test_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Removed: {filename}")

if __name__ == "__main__":
    main() 