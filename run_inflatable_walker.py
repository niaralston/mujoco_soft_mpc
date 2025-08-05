#!/usr/bin/env python3
"""
Test script for Inflatable Walker with flexible components
"""

import mujoco
import numpy as np
import time
import pathlib

def test_inflatable_walker():
    """Test the inflatable walker model with flexible components"""
    
    # Path to the inflatable walker model
    model_path = "mjpc/tasks/softwalker/inflatable_walker.xml"
    
    try:
        print("Loading inflatable walker model...")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print("✓ Model loaded successfully!")
        print(f"  - Number of bodies: {model.nbody}")
        print(f"  - Number of joints: {model.njnt}")
        print(f"  - Number of actuators: {model.nu}")
        print(f"  - Number of flex components: {model.nflex}")
        
        if model.nflex > 0:
            print("✓ Flexible components detected!")
            print(f"  - Flex elements: {model.nflex}")
            print(f"  - Flex vertices: {model.nflexvert}")
            print(f"  - Flex edges: {model.nflexedge}")
        else:
            print("⚠ No flexible components found in the model")
        
        # Create renderer for visualization
        renderer = mujoco.Renderer(model)
        
        print("\nStarting simulation...")
        print("Simulation will run for 5 seconds...")
        
        # Simulation parameters
        T = 2500  # Number of steps (5 seconds at 0.002 timestep)
        
        # Initialize data
        mujoco.mj_resetData(model, data)
        
        # Set initial position (slightly off balance to make it interesting)
        if model.nq > 2:
            data.qpos[2] = 0.1  # Small initial tilt
        
        # Simulation loop
        for t in range(T):
            if t % 500 == 0:
                print(f"Step {t}/{T}")
            
            # Apply simple control - sinusoidal torques
            time_val = data.time
            
            for i in range(model.nu):
                if i % 2 == 0:  # Hip joints
                    data.ctrl[i] = 0.3 * np.sin(time_val * 1.5)
                else:  # Knee joints
                    data.ctrl[i] = 0.2 * np.sin(time_val * 1.5 + np.pi/2)
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Render every 10th frame to save memory
            if t % 10 == 0:
                renderer.update_scene(data)
        
        print("✓ Simulation completed!")
        
        # Display some statistics
        print(f"\nSimulation Statistics:")
        print(f"  - Final time: {data.time:.2f} seconds")
        print(f"  - Final position: {data.qpos}")
        print(f"  - Final velocity: {data.qvel}")
        
        # Save a frame from the end
        renderer.update_scene(data)
        pixels = renderer.render()
        
        # Try to display the final frame if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(pixels)
            plt.title("Final Frame of Inflatable Walker Simulation")
            plt.axis('off')
            plt.show()
            print("✓ Final frame displayed!")
        except ImportError:
            print("Matplotlib not available - cannot display final frame")
            print("Final frame rendered successfully but not displayed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Inflatable Walker with Flexible Components")
    print("=" * 50)
    
    success = test_inflatable_walker()
    
    if not success:
        print("\n❌ Failed to run simulation. Check the error messages above.") 