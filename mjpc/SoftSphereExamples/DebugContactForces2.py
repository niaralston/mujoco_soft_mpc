import mujoco
import mujoco.viewer
import time
import os
import sys
import cv2  # For video recording
import math  # For converting radians to degrees
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- User Settings ---
RECORD_VIDEO = False  # Set to True to record the simulation, False to just view it
TARGET_FPS = 15     # Even lower FPS for guaranteed smooth playback
PLAYBACK_SPEED = 0.5  # Slow motion for better visualization

def get_videos_dir():
    """Create and return a directory for saving videos"""
    # Try to use Documents folder first
    try:
        documents_dir = os.path.expanduser('~/Documents')
        videos_dir = os.path.join(documents_dir, 'MuJoCo_Videos')
    except:
        # Fallback to current directory
        videos_dir = os.path.join(os.getcwd(), 'videos')
    
    # Create directory if it doesn't exist
    os.makedirs(videos_dir, exist_ok=True)
    return videos_dir

def analyze_contact_forces(times, com_positions, contact_forces, foot_positions, net_forces_list, net_torques_list):
    """Analyze the collected contact force data."""
    print("\n=== CONTACT FORCE ANALYSIS ===")
    
    # Check COM movement
    if len(com_positions) > 1:
        com_x_start = com_positions[0][0]
        com_x_end = com_positions[-1][0]
        com_x_change = com_x_end - com_x_start
        
        print(f"COM x position change: {com_x_change:.4f}")
        print(f"  Start: {com_x_start:.4f}")
        print(f"  End: {com_x_end:.4f}")
        
        if com_x_change < 0:
            print("⚠️  Robot is moving in NEGATIVE x direction!")
        else:
            print("✅ Robot is moving in POSITIVE x direction")
    
    # Analyze contact patterns
    total_contacts = sum(len(cf) for cf in contact_forces)
    print(f"\nTotal floor contacts detected: {total_contacts}")
    
    if total_contacts > 0:
        # Find average contact force direction
        all_forces = []
        for cf_list in contact_forces:
            for cf in cf_list:
                if cf['force'] > 0.01:  # Only significant forces
                    all_forces.append(cf)
        
        if all_forces:
            avg_force_x = np.mean([cf['force'] * cf['normal'][0] for cf in all_forces])
            print(f"Average contact force in x direction: {avg_force_x:.4f}")
            
            if avg_force_x < 0:
                print("⚠️  Contact forces are pushing robot in negative x direction")
            else:
                print("✅ Contact forces are pushing robot in positive x direction")
        
        # Analyze net forces over time
        if len(net_forces_list) > 0:
            try:
                net_forces_array = np.array(net_forces_list)
                net_torques_array = np.array(net_torques_list) if len(net_torques_list) > 0 else None
                
                print(f"\n=== NET FORCE ANALYSIS ===")
                
                # Average net force over the simulation
                avg_net_force = np.mean(net_forces_array, axis=0)
                print(f"Average net force (x,y,z): [{avg_net_force[0]:.4f}, {avg_net_force[1]:.4f}, {avg_net_force[2]:.4f}]")
                
                # Check if there's a consistent bias in x direction
                x_forces = net_forces_array[:, 0]
                positive_x_count = np.sum(x_forces > 0.01)  # Significant positive forces
                negative_x_count = np.sum(x_forces < -0.01)  # Significant negative forces
                
                print(f"X-direction force bias:")
                print(f"  Positive x forces: {positive_x_count} timesteps")
                print(f"  Negative x forces: {negative_x_count} timesteps")
                
                if negative_x_count > positive_x_count:
                    print("⚠️  More negative x forces detected - this explains backward motion!")
                elif positive_x_count > negative_x_count:
                    print("✅ More positive x forces detected - should move forward")
                else:
                    print("⚖️  Balanced x forces - no clear bias")
                
                # Check for any large individual forces
                max_force_magnitude = np.max(np.linalg.norm(net_forces_array, axis=1))
                print(f"Maximum net force magnitude: {max_force_magnitude:.4f} N")
                
                # Check if forces are balanced between feet
                if len(contact_forces) > 0 and any(contact_forces):
                    # Look at a sample of contacts to see foot distribution
                    sample_contacts = []
                    for cf_list in contact_forces:
                        if cf_list:
                            sample_contacts.extend(cf_list[:3])  # Take first 3 contacts from each timestep
                            break
                    
                    if sample_contacts:
                        left_foot_contacts = [c for c in sample_contacts if "left" in c['geom1'] or "left" in c['geom2']]
                        right_foot_contacts = [c for c in sample_contacts if "right" in c['geom1'] or "right" in c['geom2']]
                        
                        print(f"\nFoot contact distribution:")
                        print(f"  Left foot contacts: {len(left_foot_contacts)}")
                        print(f"  Right foot contacts: {len(right_foot_contacts)}")
                        
                        if abs(len(left_foot_contacts) - len(right_foot_contacts)) > 2:
                            print("⚠️  Uneven contact distribution between feet - may cause asymmetric forces")
            except Exception as e:
                print(f"Warning: Could not analyze net forces: {e}")
        else:
            print("\n=== NET FORCE ANALYSIS ===")
            print("No net force data available - robot may not be in contact with ground")

def plot_results(times, com_positions, contact_forces, foot_positions, net_forces_list, net_torques_list):
    """Plot the analysis results."""
    
    # Determine number of subplots based on available data
    if net_forces_list and len(net_forces_list) > 0:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Softwalker Contact Force Analysis')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Softwalker Contact Force Analysis (No Net Force Data)')
        # Convert single axes to 2D array for consistency
        if not hasattr(axes, '__len__') or len(axes) == 2:
            axes = np.array(axes).reshape(2, 2)
        else:
            axes = axes.reshape(2, 2)
    
    # COM position over time
    com_x = [pos[0] for pos in com_positions]
    com_y = [pos[1] for pos in com_positions]
    com_z = [pos[2] for pos in com_positions]
    
    axes[0, 0].plot(times, com_x, label='X', linewidth=2)
    axes[0, 0].plot(times, com_y, label='Y', linewidth=2)
    axes[0, 0].plot(times, com_z, label='Z', linewidth=2)
    axes[0, 0].set_title('Center of Mass Position')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Contact force magnitude over time
    contact_magnitudes = [sum(cf['force'] for cf in cf_list) if cf_list else 0 for cf_list in contact_forces]
    axes[0, 1].plot(times, contact_magnitudes, 'r-', linewidth=2)
    axes[0, 1].set_title('Total Contact Force Magnitude')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Force (N)')
    axes[0, 1].grid(True)
    
    # Foot positions
    right_foot_x = [pos[0][0] for pos in foot_positions]
    left_foot_x = [pos[1][0] for pos in foot_positions]
    
    axes[1, 0].plot(times, right_foot_x, label='Right Foot', linewidth=2)
    axes[1, 0].plot(times, left_foot_x, label='Left Foot', linewidth=2)
    axes[1, 0].set_title('Foot X Positions')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('X Position (m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # COM velocity (derivative of position)
    com_vel_x = np.gradient(com_x, times)
    axes[1, 1].plot(times, com_vel_x, 'g-', linewidth=2)
    axes[1, 1].set_title('COM X Velocity')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Velocity (m/s)')
    axes[1, 1].grid(True)
    
    # Net forces over time (only if we have 3x2 subplot layout)
    if net_forces_list and len(net_forces_list) > 0 and len(axes.shape) > 1 and axes.shape[0] > 2:
        net_forces_array = np.array(net_forces_list)
        
        # Check if we have net_torques data
        if net_torques_list and len(net_torques_list) > 0:
            net_torques_array = np.array(net_torques_list)
            
            # Net force components
            axes[2, 0].plot(times, net_forces_array[:, 0], 'r-', label='X', linewidth=2)
            axes[2, 0].plot(times, net_forces_array[:, 1], 'g-', label='Y', linewidth=2)
            axes[2, 0].plot(times, net_forces_array[:, 2], 'b-', label='Z', linewidth=2)
            axes[2, 0].set_title('Net Contact Force on Robot')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_ylabel('Force (N)')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Net torque components
            axes[2, 1].plot(times, net_torques_array[:, 0], 'r-', label='X', linewidth=2)
            axes[2, 1].plot(times, net_torques_array[:, 1], 'g-', label='Y', linewidth=2)
            axes[2, 1].plot(times, net_torques_array[:, 2], 'b-', label='Z', linewidth=2)
            axes[2, 1].set_title('Net Contact Torque on Robot')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Torque (N⋅m)')
            axes[2, 1].grid(True)
            axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        else:
            # Just plot net forces if no torque data
            axes[2, 0].plot(times, net_forces_array[:, 0], 'r-', label='X', linewidth=2)
            axes[2, 0].plot(times, net_forces_array[:, 1], 'g-', label='Y', linewidth=2)
            axes[2, 0].plot(times, net_forces_array[:, 2], 'b-', label='Z', linewidth=2)
            axes[2, 0].set_title('Net Contact Force on Robot')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_ylabel('Force (N)')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # No torque data available
            axes[2, 1].text(0.5, 0.5, 'No torque data available', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Net Contact Torque on Robot')
            axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    global RECORD_VIDEO  # Declare RECORD_VIDEO as global
    
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Python version: {sys.version}")
    
    # Set the model path to test_softwalker_radial.xml
    model_path = "test_softwalker_trillinear.xml"
    
    print(f"Loading model from: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: File '{model_path}' not found!")
        print("Make sure you're running this script from the SoftSphereExamples directory")
        return
    
    try:
        # Try to set plugin directory
        plugin_dir = os.path.join(os.path.dirname(mujoco.__file__), "plugin")
        if os.path.exists(plugin_dir):
            os.environ["MUJOCO_PLUGIN_DIR"] = plugin_dir
            print(f"✅ Set MUJOCO_PLUGIN_DIR to: {plugin_dir}")
        else:
            binary_plugin_dir = r"C:\mujoco-3.3.2-windows-x86_64\bin\mujoco_plugin"
            if os.path.exists(binary_plugin_dir):
                os.environ["MUJOCO_PLUGIN_DIR"] = binary_plugin_dir
                print(f"✅ Set MUJOCO_PLUGIN_DIR to: {binary_plugin_dir}")
        
        # Load the model
        print(f"\nLoading model from {model_path}...")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("✅ Successfully loaded the model!")
        
        # Print model information
        print(f"Number of bodies: {model.nbody}")
        print(f"Number of geoms: {model.ngeom}")
        print(f"Timestep: {model.opt.timestep}")
        
        # Capacity diagnostics (from <size>):
        try:
            # mjtNum is typically 8-byte float; show approx KB
            approx_kb = model.nstack * 8 / 1024.0
            print(f"nstack: {model.nstack} (~{approx_kb:.1f} KB if 8-byte mjtNum)")
        except Exception:
            print(f"nstack: {getattr(model, 'nstack', 'N/A')}")
        print(f"nconmax: {getattr(model, 'nconmax', 'N/A')}  njmax: {getattr(model, 'njmax', 'N/A')}")
        if hasattr(model.opt, 'iterations'):
            print(f"solver iterations: {model.opt.iterations}  ls_iterations: {getattr(model.opt, 'ls_iterations', 'N/A')}")
        
        # Check for flex components
        try:
            flex_elements = hasattr(model, 'nflex') and model.nflex > 0
            if flex_elements:
                print(f"✅ Model has {model.nflex} flex elements!")
        except:
            pass
        
        # Set simulation time to 60 seconds (no user input)
        sim_time = 60
        print(f"\nSimulation will run for {sim_time} seconds")
        
        # Run the simulation
        print("\nStarting simulation...")
        print("This will open a viewer window. Close the window to end the simulation.")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera to a reasonable distance
            viewer.cam.distance = 5.0
            
            # Set up simulation parameters
            fps = 30
            timestep = model.opt.timestep
            steps_per_frame = int(1.0 / (fps * timestep))
            
            # Video recording setup
            renderer = None
            if RECORD_VIDEO:
                # Use a resolution that works with default framebuffer
                width, height = 640, 480
                
                # Get the base name of the XML file without extension
                base_name = os.path.splitext(os.path.basename(model_path))[0]
                video_filename = f'{base_name}.mp4'
                
                # Set up video file path
                video_dir = os.path.dirname(os.path.abspath(__file__))
                video_path = os.path.join(video_dir, video_filename)
                print(f"\nVideo will be saved to: {video_path}")
                
                # Create renderer for video
                renderer = mujoco.Renderer(model, height=height, width=width)
                renderer.update_scene(data, camera=viewer.cam)
                
                # Use H264 codec for MP4
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)
                
                if not out.isOpened():
                    raise RuntimeError("Failed to create video writer. Check if you have write permissions in this directory.")
                
                print("Recording simulation... (Move camera with mouse to adjust view)")
                frames_written = 0
            
            print("\nRunning simulation...")
            step_count = 0
            
            # Track initial torso world position
            torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            initial_torso_pos = data.xpos[torso_id].copy()
            print(f"\nInitial torso world position - X: {initial_torso_pos[0]:.3f}, Y: {initial_torso_pos[1]:.3f}, Z: {initial_torso_pos[2]:.3f}")
            print(f"Initial joint positions (rootz, rootx, rooty) - {data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {math.degrees(data.qpos[2]):.1f}°")
            
            # Initialize contact force analysis data storage
            times = []
            com_positions = []
            contact_forces = []
            foot_positions = []
            net_forces_list = []
            net_torques_list = []
            
            print("\nStarting contact force analysis...")
            print("Press 'C' in the viewer to show contact forces")
            print("Press 'F' to show force vectors")
            print("Press 'V' to show contact points")
            
            while viewer.is_running() and data.time < sim_time:
                # Run fixed number of steps
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                    step_count += 1
                
                # Update viewer and record frame
                viewer.sync()
                
                # Store data for contact force analysis
                time_val = data.time
                times.append(time_val)
                
                # Center of mass position
                com_pos = data.xipos[0].copy()  # torso COM
                com_positions.append(com_pos.copy())
                
                # Contact forces analysis
                contact_force_data = []
                net_force = np.zeros(3)  # [x, y, z] net force on robot
                net_torque = np.zeros(3)  # Net torque around robot COM
                
                # Get constraint forces after stepping
                for j in range(data.ncon):
                    contact = data.contact[j]
                    # Check if this is a floor contact by looking at the geoms involved
                    geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom{contact.geom1}"
                    geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom{contact.geom2}"
                    
                    if "floor" in geom1_name or "floor" in geom2_name:
                        # This is a floor contact
                        pos = contact.pos
                        normal = contact.frame[:3]
                        
                        # Get the force magnitude from the constraint solver
                        # Compose contact force vector from normal and tangential components
                        try:
                            force_vector = np.zeros(3)
                            if hasattr(data, 'efc_force') and data.efc_force is not None:
                                addr = getattr(contact, 'efc_address', -1)
                                dim = getattr(contact, 'dim', 1)
                                # Frame basis: [n, t1, t2]
                                n = contact.frame[:3]
                                t1v = contact.frame[3:6]
                                t2v = contact.frame[6:9]
                                # Add normal component
                                if addr >= 0 and addr < len(data.efc_force):
                                    fn = data.efc_force[addr]
                                    force_vector += fn * n
                                # Add tangential components if available
                                if dim > 1 and addr + 1 < len(data.efc_force):
                                    ft1 = data.efc_force[addr + 1]
                                    force_vector += ft1 * t1v
                                if dim > 2 and addr + 2 < len(data.efc_force):
                                    ft2 = data.efc_force[addr + 2]
                                    force_vector += ft2 * t2v
                            else:
                                # Fallback: approximate along normal using penetration
                                fn = max(0, -contact.dist) * 1000.0
                                force_vector = fn * normal
                        except Exception:
                            # Final fallback
                            fn = max(0, -contact.dist) * 1000.0
                            force_vector = fn * normal
                        
                        # Add to net force
                        net_force += force_vector
                        
                        # Calculate torque around robot COM (r × F)
                        r = pos - data.xipos[0]  # Vector from COM to contact point
                        torque = np.cross(r, force_vector)
                        net_torque += torque
                        
                        contact_force_data.append({
                            'force': float(np.linalg.norm(force_vector)),
                            'position': pos,
                            'normal': normal,
                            'force_vector': force_vector,
                            'geom1': geom1_name,
                            'geom2': geom2_name
                        })
                
                contact_forces.append(contact_force_data)
                
                # Store net forces for analysis
                net_forces_list.append(net_force.copy())
                net_torques_list.append(net_torque.copy())
                
                # Foot positions
                try:
                    right_foot_pos = data.site_xpos[model.site_name2id("right_foot_center")]
                    left_foot_pos = data.site_xpos[model.site_name2id("left_foot_center")]
                    foot_positions.append([right_foot_pos.copy(), left_foot_pos.copy()])
                except:
                    # Fallback to torso position if sites not found
                    foot_positions.append([data.xipos[0].copy(), data.xipos[0].copy()])
                
                # Print position every second (adjust fps * 1 for frequency)
                if step_count % (fps * 1) == 0:  # Print every second
                    # Get current torso world position
                    current_torso_pos = data.xpos[torso_id]
                    torso_dx = current_torso_pos[0] - initial_torso_pos[0]
                    torso_dy = current_torso_pos[1] - initial_torso_pos[1]
                    torso_dz = current_torso_pos[2] - initial_torso_pos[2]
                    
                    # Also print joint positions for comparison
                    joint_x = data.qpos[0]  # rootz
                    joint_y = data.qpos[1]  # rootx  
                    joint_rot = data.qpos[2]  # rooty (rotation)
                    
                    print(f"Time: {data.time:.2f}s")
                    print(f"  Torso world pos: ({current_torso_pos[0]:.3f}, {current_torso_pos[1]:.3f}, {current_torso_pos[2]:.3f})")
                    print(f"  Torso delta: ({torso_dx:.3f}, {torso_dy:.3f}, {torso_dz:.3f})")
                    print(f"  Joint pos (z,x,rot): ({joint_x:.3f}, {joint_y:.3f}, {math.degrees(joint_rot):.1f}°)")
                    print(f"  Contacts: {data.ncon}")
                    print(f"  Net Force (x,y,z): [{net_force[0]:.4f}, {net_force[1]:.4f}, {net_force[2]:.4f}]")
                    print(f"  Net Torque (x,y,z): [{net_torque[0]:.4f}, {net_torque[1]:.4f}, {net_torque[2]:.4f}]")
                    
                    if contact_force_data:
                        print(f"  Active contacts: {len(contact_force_data)}")
                        # Group contacts by foot for better analysis
                        left_foot_forces = []
                        right_foot_forces = []
                        
                        for contact in contact_force_data:
                            # Determine which foot this contact belongs to
                            if "left" in contact['geom1'] or "left" in contact['geom2']:
                                left_foot_forces.append(contact)
                            elif "right" in contact['geom1'] or "right" in contact['geom2']:
                                right_foot_forces.append(contact)
                            
                            print(f"    {contact['geom1']} <-> {contact['geom2']}: Force={contact['force']:.3f}, Vector=[{contact['force_vector'][0]:.3f}, {contact['force_vector'][1]:.3f}, {contact['force_vector'][2]:.3f}]")
                        
                        # Calculate net force per foot
                        if left_foot_forces:
                            left_net = np.sum([cf['force_vector'] for cf in left_foot_forces], axis=0)
                            print(f"  Left foot net force: [{left_net[0]:.4f}, {left_net[1]:.4f}, {left_net[2]:.4f}]")
                        
                        if right_foot_forces:
                            right_net = np.sum([cf['force_vector'] for cf in right_foot_forces], axis=0)
                            print(f"  Right foot net force: [{right_net[0]:.4f}, {right_net[1]:.4f}, {right_net[2]:.4f}]")
                    else:
                        print("  No floor contacts detected")
                    
                    # Debug: Print contact details for first few contacts
                    if step_count % (fps * 5) == 0 and data.ncon > 0:  # Every 5 seconds
                        print(f"    Contact details (first 5 of {data.ncon}):")
                        for i in range(min(5, data.ncon)):
                            contact = data.contact[i]
                            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom{contact.geom1}"
                            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom{contact.geom2}"
                            print(f"      {geom1_name} <-> {geom2_name}, dist: {contact.dist:.4f}")
                
                if RECORD_VIDEO and renderer:
                    renderer.update_scene(data, camera=viewer.cam)
                    pixels = renderer.render()
                    out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                    frames_written += 1
                
                # Print progress occasionally
                if step_count % (fps * 10) == 0:  # Every 10 seconds
                    print(f"Simulation time: {data.time:.2f}s / {sim_time:.2f}s")
                
                # Small sleep to keep the simulation visible
                time.sleep(0.01)
            
            # Clean up
            print(f"\nSimulation finished. Simulation time: {data.time:.2f}s")
            if RECORD_VIDEO:
                print(f"Frames written: {frames_written}")
                print("Saving video...")
                out.release()
                
                # Verify the video file was created
                if os.path.exists(video_path):
                    print(f"Video saved successfully to: {video_path}")
                    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
                else:
                    print("Warning: Video file was not created successfully!")
        
        # Analyze the collected data
        analyze_contact_forces(times, com_positions, contact_forces, foot_positions, net_forces_list, net_torques_list)
        
        # Plot the results
        plot_results(times, com_positions, contact_forces, foot_positions, net_forces_list, net_torques_list)
        
        print("\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 