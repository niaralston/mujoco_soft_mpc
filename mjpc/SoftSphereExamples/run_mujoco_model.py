import mujoco
import mujoco.viewer
import time
import os
import sys
import cv2  # For video recording
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

def main():
    global RECORD_VIDEO  # Declare RECORD_VIDEO as global
    
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Python version: {sys.version}")
    
    # Look for XML files in current directory, Examples subdirectory, and Models2/Deepmind directory
    xml_files = []
    
    # Check current directory
    current_dir_files = [f for f in os.listdir('.') if f.endswith('.xml')]
    for file in current_dir_files:
        xml_files.append(file)
    
    # Check Examples subdirectory if it exists
    examples_dir = os.path.join('.', 'Examples')
    if os.path.exists(examples_dir) and os.path.isdir(examples_dir):
        examples_files = [os.path.join('Examples', f) for f in os.listdir(examples_dir) if f.endswith('.xml')]
        for file in examples_files:
            xml_files.append(file)
            
    # Check Models2/Deepmind directory if it exists
    deepmind_dir = os.path.join('.', 'Models2', 'Deepmind')
    if os.path.exists(deepmind_dir) and os.path.isdir(deepmind_dir):
        deepmind_files = [os.path.join('Models2/Deepmind', f) for f in os.listdir(deepmind_dir) if f.endswith('.xml')]
        for file in deepmind_files:
            xml_files.append(file)
    
    if not xml_files:
        print("No XML files found in the current directory, Examples subdirectory, or Models2/Deepmind directory!")
        return
    
    # Display available XML files with numbers
    print("\nAvailable MuJoCo XML files:")
    for i, file in enumerate(xml_files, 1):
        print(f"{i}. {file}")
    
    # Get user selection
    while True:
        try:
            choice = input("\nEnter the number or name of the file you want to run: ")
            
            # Check if input is a number
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(xml_files):
                    model_path = xml_files[index]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(xml_files)}")
            # Check if input is a filename
            elif choice in xml_files:
                model_path = choice
                break
            else:
                # Check if user entered filename without path
                matching_files = [f for f in xml_files if f.endswith('/' + choice) or f == choice]
                if matching_files:
                    model_path = matching_files[0]
                    break
                
                # Check if user entered filename without .xml extension
                if choice + ".xml" in xml_files:
                    model_path = choice + ".xml"
                    break
                
                # Check if user entered filename without Examples/ prefix
                examples_match = os.path.join('Examples', choice)
                if examples_match in xml_files:
                    model_path = examples_match
                    break
                
                # Check if user entered filename without Examples/ prefix and .xml extension
                examples_match_ext = os.path.join('Examples', choice + ".xml")
                if examples_match_ext in xml_files:
                    model_path = examples_match_ext
                    break
                
                print("Invalid selection. Please enter a valid number or filename.")
        except ValueError:
            print("Please enter a valid number or filename.")
    
    print(f"Loading model from: {model_path}")
    
    # Check if file exists (should always be true at this point)
    if not os.path.exists(model_path):
        print(f"❌ Error: File '{model_path}' not found!")
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
        
        # Check for flex components
        try:
            flex_elements = hasattr(model, 'nflex') and model.nflex > 0
            if flex_elements:
                print(f"✅ Model has {model.nflex} flex elements!")
        except:
            pass
        
        # Ask for simulation time
        try:
            sim_time = int(input("\nEnter simulation time in seconds (default: 60): ") or "60")
        except ValueError:
            sim_time = 60
            print("Invalid input. Using default simulation time of 60 seconds.")
        
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
                video_filename = f'{base_name}.avi'
                
                # Set up video file path
                video_dir = os.path.dirname(os.path.abspath(__file__))
                video_path = os.path.join(video_dir, video_filename)
                print(f"\nVideo will be saved to: {video_path}")
                
                # Create renderer for video
                renderer = mujoco.Renderer(model, height=height, width=width)
                renderer.update_scene(data, camera=viewer.cam)
                
                # Use XVID codec which is more widely supported
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)
                
                if not out.isOpened():
                    raise RuntimeError("Failed to create video writer. Check if you have write permissions in this directory.")
                
                print("Recording simulation... (Move camera with mouse to adjust view)")
                frames_written = 0
            
            print("\nRunning simulation...")
            step_count = 0
            
            while viewer.is_running() and data.time < sim_time:
                # Run fixed number of steps
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                    step_count += 1
                
                # Update viewer and record frame
                viewer.sync()
                
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
            print(f"\nSimulation finished. Simulation time: {data.time:.2f} seconds")
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
        
        print("\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()