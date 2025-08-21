#!/usr/bin/env python3
"""
Debug script to analyze contact forces and robot movement in the softwalker simulation.
This will help identify why the robot is moving in the negative x direction.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def analyze_contact_forces(model_path):
    """Analyze contact forces and robot movement."""
    
    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Simulation parameters
    sim_time = 5.0  # seconds
    dt = model.opt.timestep
    n_steps = int(sim_time / dt)
    
    # Storage for data
    times = []
    com_positions = []
    contact_forces = []
    foot_positions = []
    
    # Initialize net force storage
    net_forces_list = []
    net_torques_list = []
    
    print("Starting simulation to analyze contact forces...")
    print("Press 'C' in the viewer to show contact forces")
    print("Press 'F' to show force vectors")
    print("Press 'V' to show contact points")
    
    # Run simulation
    for i in range(n_steps):
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Store data
        time_val = i * dt
        times.append(time_val)
        
        # Center of mass position
        com_pos = data.xipos[0].copy()  # torso COM
        com_positions.append(com_pos.copy())
        
        # Contact forces (if any contacts exist)
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
                # In MuJoCo, we need to access the constraint forces correctly
                try:
                    # Try to get force from constraint solver
                    if hasattr(data, 'efc_force') and data.efc_force is not None and j < len(data.efc_force):
                        force = data.efc_force[j]
                    else:
                        # Fallback: use contact distance as a proxy for force
                        force = max(0, -contact.dist) * 1000  # Convert distance to force estimate
                except:
                    # If all else fails, use a simple estimate
                    force = max(0, -contact.dist) * 1000
                
                # Calculate force vector (force * normal direction)
                force_vector = force * normal
                
                # Add to net force
                net_force += force_vector
                
                # Calculate torque around robot COM (r × F)
                r = pos - data.xipos[0]  # Vector from COM to contact point
                torque = np.cross(r, force_vector)
                net_torque += torque
                
                contact_force_data.append({
                    'force': force,
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
        
        # Print progress every second
        if i % int(1.0 / dt) == 0:
            print(f"Time: {time_val:.1f}s, COM x: {com_pos[0]:.3f}")
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
    
    # Analyze the data
    print("\n=== ANALYSIS RESULTS ===")
    
    # Check COM movement
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
                net_torques_array = np.array(net_torques_list)
                
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
    
    # Plot results
    net_forces_data = net_forces_list if len(net_forces_list) > 0 else None
    print(f"Debug: Found {len(net_forces_list)} net force entries")
    
    plot_results(times, com_positions, contact_forces, foot_positions, data, net_forces_data, net_torques_list)
    
    return data

def plot_results(times, com_positions, contact_forces, foot_positions, data=None, net_forces=None, net_torques=None):
    """Plot the analysis results."""
    
    # Determine number of subplots based on available data
    if net_forces is not None and len(net_forces) > 0:
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
    
    # Net forces over time
    if net_forces is not None and len(net_forces) > 0 and len(axes.shape) > 1 and axes.shape[0] > 2:
        net_forces_array = np.array(net_forces)
        
        # Check if we have net_torques data
        if net_torques is not None and len(net_torques) > 0:
            net_torques_array = np.array(net_torques)
            
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
    """Main function to run the analysis."""
    model_path = "test_softwalker_radial.xml"
    
    try:
        data = analyze_contact_forces(model_path)
        print("\nAnalysis complete! Check the plots for detailed information.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure you're running this script from the SoftSphereExamples directory")

if __name__ == "__main__":
    main() 