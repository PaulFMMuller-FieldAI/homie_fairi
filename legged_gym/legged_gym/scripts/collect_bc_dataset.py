
"""
Collect BC (Behavior Cloning) dataset using Isaac Gym simulation.

This script uses Isaac Gym to simulate a G1 robot, sends random velocity commands 
(x_vel, y_vel, yaw_vel), and collects trajectories for BC training. By default, it uses
Isaac Gym simulation for realistic physics-based data collection.

Note: Isaac Gym mode uses Isaac Gym's physics simulation (NOT Unitree SDK). 
      Unitree SDK is only used when --real-robot flag is specified.

Usage:
    # Collect data from Isaac Gym simulation (default)
    python legged_gym/scripts/collect_bc_dataset.py \
        --task g1 \
        --num-trajectories 100 \
        --traj-duration 10.0 \
        --frequency 50.0 \
        --output-dir ./bc_dataset \
        --headless
    
    # Collect data from real G1 robot (optional)
    python legged_gym/scripts/collect_bc_dataset.py \
        --real-robot \
        --network-interface enp3s0 \
        --num-trajectories 100 \
        --traj-duration 10.0 \
        --frequency 50.0 \
        --output-dir ./bc_dataset

Arguments:
    --task: Task name for Isaac Gym (default: g1). Required unless --real-robot is used.
    --num-trajectories: Total number of trajectories to collect (default: 100)
    --traj-duration: Duration of each trajectory in seconds (default: 10.0)
    --frequency: Sampling frequency in Hz (default: 50.0)
    --output-dir: Output directory for trajectories (default: auto-generated timestamp)
    --headless: Run Isaac Gym in headless mode (no visualization). Default: False.
    --real-robot: Use real robot instead of Isaac Gym simulation (requires network interface)
    --network-interface: Network interface connected to G1 (e.g., enp3s0, eth0). Required for --real-robot.

Output:
    Each trajectory is saved as:
    - trajectory_XXXXX.npz: Contains observations, actions, commands, and state data
    - trajectory_XXXXX_metadata.json: Metadata about the trajectory
    - dataset_metadata.json: Overall dataset metadata

The saved data includes:
    - observations: Full observation vectors (if using state estimator)
    - actions: Joint actions from Unitree controller (from LCM)
    - commands: High-level velocity commands sent to Unitree controller
    - dof_pos, dof_vel: Joint positions and velocities
    - base_lin_vel, base_ang_vel: Base linear and angular velocities
    - base_quat, base_pos: Base orientation and position
"""

import os
import sys
import numpy as np
import argparse
from datetime import datetime
import json
import time

# LCM import (only needed for real robot mode)
try:
    import lcm
except ImportError:
    lcm = None  # Will check later if needed

# Add legged_gym to path if needed (for when script is run directly)
script_dir = os.path.dirname(os.path.abspath(__file__))
legged_gym_root = os.path.join(script_dir, '../../..')
if os.path.exists(os.path.join(legged_gym_root, 'legged_gym')):
    sys.path.insert(0, legged_gym_root)

# Isaac Gym imports (must be imported before torch)
ISAAC_GYM_AVAILABLE = False
try:
    import isaacgym
    ISAAC_GYM_AVAILABLE = True
except ImportError as e:
    pass  # Will show error later if needed

if ISAAC_GYM_AVAILABLE:
    try:
        from legged_gym.envs import *
        from legged_gym.utils import get_args, task_registry
    except ImportError as e:
        ISAAC_GYM_AVAILABLE = False

import torch  # Import torch after isaacgym

# Unitree SDK imports (only needed for real robot mode)
UNITREE_SDK_AVAILABLE = False
ChannelFactoryInitialize = None
LocoClient = None

unitree_sdk2_path = os.path.join(os.path.dirname(__file__), '../../../HomieDeploy/unitree_sdk2')
if os.path.exists(unitree_sdk2_path):
    # Check for Python bindings in build directory
    build_python_path = os.path.join(unitree_sdk2_path, 'build', 'python')
    if os.path.exists(build_python_path):
        sys.path.insert(0, build_python_path)
    # Also try adding the build directory itself
    build_path = os.path.join(unitree_sdk2_path, 'build')
    if os.path.exists(build_path):
        sys.path.insert(0, build_path)

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    pass  # Will check later if needed for real robot mode

# State estimator for getting robot state (only needed for real robot mode)
StateEstimator = None
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../HomieDeploy/g1_gym_deploy'))
    from utils.cheetah_state_estimator import StateEstimator
    from lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
    from lcm_types.arm_action_lcmt import arm_action_lcmt
except ImportError:
    pass  # Will check later if needed for real robot mode


def generate_random_commands():
    """
    Generate random velocity commands similar to Unitree high-level controller input.
    
    Returns:
        Tuple (x_vel, y_vel, yaw_vel, height) with random values in reasonable ranges
    """
    x_vel = np.random.uniform(-0.8, 1.2)  # Forward/backward velocity
    y_vel = np.random.uniform(-0.5, 0.5)  # Lateral velocity
    yaw_vel = np.random.uniform(-0.8, 0.8)  # Yaw velocity
    height = np.random.uniform(0.68, 0.74)  # Standing height
    
    return x_vel, y_vel, yaw_vel, height

def smooth_command_transition(old_cmd, new_cmd, alpha=0.3):
    """
    Smoothly transition between commands to avoid abrupt changes.
    
    Args:
        old_cmd: Previous command (x_vel, y_vel, yaw_vel, height)
        new_cmd: New command (x_vel, y_vel, yaw_vel, height)
        alpha: Smoothing factor (0 = no change, 1 = instant change)
    
    Returns:
        Smoothed command
    """
    return [
        old_cmd[0] * (1 - alpha) + new_cmd[0] * alpha,
        old_cmd[1] * (1 - alpha) + new_cmd[1] * alpha,
        old_cmd[2] * (1 - alpha) + new_cmd[2] * alpha,
        old_cmd[3] * (1 - alpha) + new_cmd[3] * alpha,
    ]

def prepare_robot(loco_client: LocoClient):
    """Prepare robot for data collection."""
    print("[1/3] Setting robot into damp mode...")
    loco_client.Damp()
    time.sleep(2.0)
    
    print("[2/3] Setting robot into start mode...")
    loco_client.Start()
    time.sleep(5.0)
    
    print("[3/3] Robot ready for data collection!")

def cleanup_robot(loco_client: LocoClient):
    """Clean up robot after data collection."""
    print("Cleaning up robot...")
    loco_client.StopMove()
    time.sleep(1.0)
    loco_client.Damp()
    time.sleep(2.0)

def collect_trajectory_isaac_gym(env, traj_idx: int, output_dir: str, 
                                  traj_duration: float = 10.0, frequency: float = 50.0):
    """
    Collect a single trajectory from Isaac Gym simulation.
    
    Args:
        env: Isaac Gym environment
        traj_idx: Trajectory index
        output_dir: Output directory
        traj_duration: Duration in seconds
        frequency: Sampling frequency in Hz
    
    Returns:
        True if successful, False otherwise
    """
    dt = 1.0 / frequency
    num_steps = int(traj_duration / dt)
    sim_dt = env.dt  # Isaac Gym simulation timestep
    steps_per_sample = max(1, int(sim_dt * frequency))  # How many sim steps per sample
    
    # Initialize trajectory storage
    trajectory = {
        'observations': [],
        'actions': [],
        'commands': [],
        'dof_pos': [],
        'dof_vel': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
        'base_quat': [],
        'base_pos': [],
        'timestamps': [],
    }
    
    # Reset environment
    env.reset()
    obs = env.get_observations()
    
    # Generate initial command
    current_command = generate_random_commands()
    command_change_interval = int(1.0 / dt)  # Change command every second
    command_change_counter = 0
    
    print(f"  Collecting trajectory {traj_idx} ({traj_duration}s, {num_steps} steps)...")
    
    try:
        step = 0
        sim_step = 0
        
        while step < num_steps:
            # Update command periodically
            if command_change_counter >= command_change_interval:
                new_command = generate_random_commands()
                current_command = smooth_command_transition(current_command, new_command, alpha=0.3)
                command_change_counter = 0
            
            # Set commands in environment (for env_id 0, since we use num_envs=1)
            env.commands[0, 0] = current_command[0]  # x_vel
            env.commands[0, 1] = current_command[1]  # y_vel
            env.commands[0, 2] = current_command[2]  # yaw_vel
            env.commands[0, 4] = current_command[3]  # height
            
            # Get random actions (or use a policy if available)
            # For BC dataset collection, we'll use random actions within limits
            actions = torch.rand(1, env.num_actions, device=env.device) * 2.0 - 1.0  # [-1, 1]
            
            # Step environment
            obs, _, _, _, _, _, _ = env.step(actions)
            
            # Sample data at specified frequency
            if sim_step % steps_per_sample == 0:
                # Extract state from environment (env_id 0)
                dof_pos = env.dof_pos[0].cpu().numpy()
                dof_vel = env.dof_vel[0].cpu().numpy()
                base_quat = env.base_quat[0].cpu().numpy()
                base_lin_vel = env.base_lin_vel[0].cpu().numpy()
                base_ang_vel = env.base_ang_vel[0].cpu().numpy()
                base_pos = env.root_states[0, 0:3].cpu().numpy()
                action = actions[0].cpu().numpy()
                obs_np = obs[0].cpu().numpy()
                
                # Store data
                trajectory['commands'].append(np.array(current_command))
                trajectory['dof_pos'].append(dof_pos.copy())
                trajectory['dof_vel'].append(dof_vel.copy())
                trajectory['base_ang_vel'].append(base_ang_vel.copy())
                trajectory['base_quat'].append(base_quat.copy())
                trajectory['base_pos'].append(base_pos.copy())
                trajectory['base_lin_vel'].append(base_lin_vel.copy())
                trajectory['actions'].append(action.copy())
                trajectory['observations'].append(obs_np.copy())
                trajectory['timestamps'].append(time.time())
                
                command_change_counter += 1
                step += 1
            
            sim_step += 1
            
            # Check if episode ended (reset if needed)
            if env.reset_buf[0] > 0:
                env.reset()
                obs = env.get_observations()
                # Generate new command after reset
                current_command = generate_random_commands()
                command_change_counter = 0
                sim_step = 0  # Reset sim step counter
        
        # Convert lists to numpy arrays
        traj_data = {}
        for key in trajectory:
            if len(trajectory[key]) > 0:
                traj_data[key] = np.array(trajectory[key])
        
        # Save trajectory
        traj_filename = os.path.join(output_dir, f"trajectory_{traj_idx:05d}.npz")
        np.savez_compressed(traj_filename, **traj_data)
        
        # Save metadata
        metadata = {
            'traj_idx': traj_idx,
            'duration': traj_duration,
            'frequency': frequency,
            'dt': dt,
            'num_steps': len(traj_data['observations']),
            'obs_dim': traj_data['observations'].shape[1] if len(traj_data['observations'].shape) > 1 else 1,
            'action_dim': traj_data['actions'].shape[1] if len(traj_data['actions'].shape) > 1 else 1,
            'command_dim': traj_data['commands'].shape[1] if len(traj_data['commands'].shape) > 1 else 1,
            'timestamp': datetime.now().isoformat(),
            'source': 'isaac_gym',
        }
        metadata_filename = os.path.join(output_dir, f"trajectory_{traj_idx:05d}_metadata.json")
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Trajectory {traj_idx} saved successfully")
        return True
        
    except Exception as e:
        print(f"  Error collecting trajectory {traj_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False

def collect_trajectory(loco_client: LocoClient, state_estimator: StateEstimator, 
                       traj_idx: int, output_dir: str, traj_duration: float = 10.0, 
                       frequency: float = 50.0):
    """
    Collect a single trajectory using Unitree's high-level controller.
    
    Args:
        loco_client: Unitree locomotion client
        state_estimator: State estimator for getting robot state
        traj_idx: Trajectory index
        output_dir: Output directory
        traj_duration: Duration in seconds
        frequency: Sampling frequency in Hz
    
    Returns:
        True if successful, False otherwise
    """
    dt = 1.0 / frequency
    num_steps = int(traj_duration / dt)
    
    # Initialize trajectory storage
    trajectory = {
        'observations': [],
        'actions': [],
        'commands': [],
        'dof_pos': [],
        'dof_vel': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
        'base_quat': [],
        'base_pos': [],
        'timestamps': [],
    }
    
    # Generate initial command
    current_command = generate_random_commands()
    command_change_interval = int(1.0 / dt)  # Change command every second
    command_change_counter = 0
    
    print(f"  Collecting trajectory {traj_idx} ({traj_duration}s, {num_steps} steps)...")
    
    try:
        start_time = time.time()
        step = 0
        
        while step < num_steps:
            # Update command periodically
            if command_change_counter >= command_change_interval:
                new_command = generate_random_commands()
                current_command = smooth_command_transition(current_command, new_command, alpha=0.3)
                command_change_counter = 0
                
                # Send velocity command to Unitree controller
                loco_client.Move(
                    current_command[0],  # x_vel
                    current_command[1],  # y_vel
                    current_command[2],  # yaw_vel
                    continous_move=False
                )
            else:
                command_change_counter += 1
            
            # Handle LCM messages to get latest state (skip if using mock)
            if state_estimator.lc is not None:
                lc = state_estimator.lc
                lc.handle_timeout(0)  # Non-blocking
            
            # Get robot state from state estimator
            dof_pos = state_estimator.get_dof_pos()
            dof_vel = state_estimator.get_dof_vel()
            rpy = state_estimator.get_rpy()
            body_ang_vel = state_estimator.get_body_angular_vel()
            
            # Convert RPY to quaternion
            roll, pitch, yaw = rpy
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            quat = np.array([
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy
            ])
            
            # Estimate base linear velocity (simplified - you may want to use state estimator's vWorld)
            # For now, we'll use zeros or estimate from IMU
            base_lin_vel = np.zeros(3)  # TODO: Get from state estimator if available
            
            # Store data
            trajectory['commands'].append(np.array(current_command))
            trajectory['dof_pos'].append(dof_pos.copy())
            trajectory['dof_vel'].append(dof_vel.copy())
            trajectory['base_ang_vel'].append(body_ang_vel.copy())
            trajectory['base_quat'].append(quat.copy())
            trajectory['base_pos'].append(np.zeros(3))  # TODO: Get from state estimator if available
            trajectory['base_lin_vel'].append(base_lin_vel.copy())
            trajectory['timestamps'].append(time.time())
            
            # TODO: Get actions from LCM if available
            # For now, we'll store zeros (actions are internal to Unitree controller)
            trajectory['actions'].append(np.zeros(len(dof_pos)))
            
            # TODO: Construct observations if needed
            # For now, we'll store a simple observation vector
            obs = np.concatenate([
                dof_pos,
                dof_vel,
                rpy,
                body_ang_vel,
                current_command[:3]  # x_vel, y_vel, yaw_vel
            ])
            trajectory['observations'].append(obs)
            
            # Sleep to maintain frequency
            elapsed = time.time() - start_time
            target_time = step * dt
            sleep_time = target_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            step += 1
        
        # Convert lists to numpy arrays
        traj_data = {}
        for key in trajectory:
            if len(trajectory[key]) > 0:
                traj_data[key] = np.array(trajectory[key])
        
        # Save trajectory
        traj_filename = os.path.join(output_dir, f"trajectory_{traj_idx:05d}.npz")
        np.savez_compressed(traj_filename, **traj_data)
        
        # Save metadata
        metadata = {
            'traj_idx': traj_idx,
            'duration': traj_duration,
            'frequency': frequency,
            'dt': dt,
            'num_steps': len(traj_data['observations']),
            'obs_dim': traj_data['observations'].shape[1] if len(traj_data['observations'].shape) > 1 else 1,
            'action_dim': traj_data['actions'].shape[1] if len(traj_data['actions'].shape) > 1 else 1,
            'command_dim': traj_data['commands'].shape[1] if len(traj_data['commands'].shape) > 1 else 1,
            'timestamp': datetime.now().isoformat(),
        }
        metadata_filename = os.path.join(output_dir, f"trajectory_{traj_idx:05d}_metadata.json")
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Trajectory {traj_idx} saved successfully")
        return True
        
    except Exception as e:
        print(f"  Error collecting trajectory {traj_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_available_interfaces():
    """Get list of available network interfaces."""
    import subprocess
    try:
        result = subprocess.run(['ip', '-o', 'link', 'show'], capture_output=True, text=True)
        interfaces = []
        for line in result.stdout.split('\n'):
            if ':' in line:
                iface = line.split(':')[1].strip()
                if iface and iface != 'lo':
                    interfaces.append(iface)
        return interfaces
    except Exception:
        return []

def validate_network_interface(interface_name):
    """Validate that a network interface exists."""
    if interface_name is None or interface_name == "":
        return True, None  # Empty string is valid (default/auto)
    
    available = get_available_interfaces()
    if interface_name in available:
        return True, None
    
    # Interface doesn't exist - provide helpful error
    error_msg = f"\nERROR: Network interface '{interface_name}' does not exist.\n"
    error_msg += "\nAvailable network interfaces:\n"
    for iface in available:
        error_msg += f"  - {iface}\n"
    
    # Suggest similar interfaces
    if interface_name.startswith('eth'):
        similar = [iface for iface in available if iface.startswith('en')]
        if similar:
            error_msg += f"\nSuggested alternatives (Ethernet interfaces):\n"
            for iface in similar[:3]:  # Show up to 3 suggestions
                error_msg += f"  - {iface}\n"
    
    error_msg += f"\nPlease specify a valid network interface with --network-interface\n"
    return False, error_msg

def main():
    parser = argparse.ArgumentParser(
        description='Collect BC dataset using Isaac Gym simulation (default) or real robot'
    )
    parser.add_argument('--task', type=str, default='g1',
                       help='Task name for Isaac Gym (default: g1)')
    parser.add_argument('--num-trajectories', type=int, default=100,
                       help='Total number of trajectories to collect')
    parser.add_argument('--traj-duration', type=float, default=10.0,
                       help='Duration of each trajectory in seconds')
    parser.add_argument('--frequency', type=float, default=50.0,
                       help='Sampling frequency in Hz')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for trajectories')
    parser.add_argument('--headless', action='store_true',
                       help='Run Isaac Gym in headless mode (no visualization)')
    parser.add_argument('--real-robot', action='store_true',
                       help='Use real robot instead of Isaac Gym simulation')
    parser.add_argument('--network-interface', type=str, default=None,
                       help='Network interface connected to G1 (required for --real-robot)')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source = "real" if args.real_robot else "sim"
        args.output_dir = f"./bc_dataset_{source}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Collecting BC dataset")
    print("=" * 60)
    if args.real_robot:
        print("  Mode: Real Robot")
        print(f"  Network interface: {args.network_interface}")
    else:
        print("  Mode: Isaac Gym Simulation")
        print(f"  Task: {args.task}")
        print(f"  Headless: {args.headless}")
    print(f"  Number of trajectories: {args.num_trajectories}")
    print(f"  Duration per trajectory: {args.traj_duration}s")
    print(f"  Sampling frequency: {args.frequency}Hz")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Use real robot if requested
    if args.real_robot:
        if not UNITREE_SDK_AVAILABLE:
            print("ERROR: Unitree SDK is required for real robot mode but is not available.")
            print("\nPlease install unitree_sdk2py Python bindings.")
            print("The unitree_sdk2 package should include Python bindings.")
            print("Check the unitree_sdk2 documentation for installation instructions.")
            print(f"\nChecked paths:")
            print(f"  - {unitree_sdk2_path}")
            sys.exit(1)
        
        # Auto-detect network interface if not provided
        network_interface = args.network_interface
        if network_interface is None:
            print("Auto-detecting network interface...")
            import subprocess
            try:
                # Get all non-loopback interfaces
                result = subprocess.run(['ip', '-o', 'link', 'show'], capture_output=True, text=True)
                interfaces = []
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        iface = line.split(':')[1].strip()
                        if iface and iface != 'lo' and not iface.startswith('docker') and not iface.startswith('tailscale'):
                            # Prefer Ethernet interfaces (en*, eth*)
                            if iface.startswith('en') or iface.startswith('eth'):
                                interfaces.insert(0, iface)
                            else:
                                interfaces.append(iface)
                
                if interfaces:
                    network_interface = interfaces[0]
                    print(f"  Using auto-detected interface: {network_interface}")
                else:
                    print("  WARNING: Could not auto-detect network interface. Trying empty string (default)...")
                    network_interface = ""
            except Exception as e:
                print(f"  WARNING: Could not auto-detect network interface ({e}). Trying empty string (default)...")
                network_interface = ""
        
        # Validate network interface before attempting initialization
        is_valid, error_msg = validate_network_interface(network_interface)
        if not is_valid:
            print(error_msg)
            sys.exit(1)
        
        # Initialize Unitree SDK
        print(f"Initializing Unitree SDK with network interface: {network_interface if network_interface else '(default/auto)'}...")
        try:
            ChannelFactoryInitialize(0, network_interface)
        except Exception as e:
            print(f"\nERROR: Failed to initialize Unitree SDK with network interface '{network_interface}'")
            print(f"Error: {e}")
            print("\nAvailable network interfaces:")
            interfaces = get_available_interfaces()
            for iface in interfaces:
                print(f"  - {iface}")
            print(f"\nPlease specify a valid network interface with --network-interface")
            print(f"Or use --test-mode to run without a real robot connection")
            print(f"Common names: enp3s0, eth0, wlan0, etc.")
            sys.exit(1)
        
        # Initialize locomotion client
        print("Connecting to G1 robot...")
        loco_client = LocoClient()
        loco_client.SetTimeout(10.0)
        loco_client.Init()
        
        # Initialize LCM and state estimator
        if lcm is None:
            print("ERROR: LCM is required for real robot mode but is not installed.")
            print("Please install LCM: pip install lcm")
            sys.exit(1)
        if StateEstimator is None:
            print("ERROR: StateEstimator is required for real robot mode but is not available.")
            print("Please ensure g1_gym_deploy is properly set up.")
            sys.exit(1)
        print("Initializing state estimator...")
        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        state_estimator = StateEstimator(lc)
        
        # Wait for first state data
        print("Waiting for robot state data...")
        timeout = 10.0
        start_wait = time.time()
        while not state_estimator.received_first_bodydate:
            lc.handle_timeout(100)  # 100ms timeout
            if time.time() - start_wait > timeout:
                raise RuntimeError("Timeout waiting for robot state data. Is the robot running?")
            time.sleep(0.1)
        print("Robot state data received!")
        
        # Prepare robot
        try:
            prepare_robot(loco_client)
        except KeyboardInterrupt:
            cleanup_robot(loco_client)
            return
        
        # Collect trajectories from real robot
        print()
        print("Starting data collection...")
        print()
        
        successful = 0
        failed = 0
        
        try:
            for traj_idx in range(args.num_trajectories):
                success = collect_trajectory(
                    loco_client,
                    state_estimator,
                    traj_idx,
                    args.output_dir,
                    traj_duration=args.traj_duration,
                    frequency=args.frequency
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                # Print progress
                if (traj_idx + 1) % 10 == 0 or traj_idx == args.num_trajectories - 1:
                    print(f"Progress: {traj_idx + 1}/{args.num_trajectories} "
                          f"({successful} successful, {failed} failed)")
                    print()
                
                # Small delay between trajectories
                if traj_idx < args.num_trajectories - 1:
                    time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cleanup_robot(loco_client)
        
        # Save dataset metadata
        dataset_metadata = {
            'num_trajectories': successful,
            'traj_duration': args.traj_duration,
            'frequency': args.frequency,
            'collection_timestamp': datetime.now().isoformat(),
            'source': 'real_robot',
            'network_interface': args.network_interface,
        }
    else:
        # Use Isaac Gym simulation (default)
        if not ISAAC_GYM_AVAILABLE:
            print("ERROR: Isaac Gym is not available!")
            print("\nTroubleshooting:")
            print("1. Make sure Isaac Gym is installed:")
            print("   cd path_to_isaac_gym/python && pip install -e .")
            print("2. Make sure you're in the correct conda environment")
            print("3. Check that isaacgym can be imported:")
            print("   python -c 'import isaacgym; print(isaacgym.__file__)'")
            print("\nAlternatively, use --real-robot to collect data from a real robot.")
            sys.exit(1)
        
        # Create Isaac Gym args
        from legged_gym.utils.helpers import get_args as get_isaac_args
        isaac_args = get_isaac_args()
        isaac_args.task = args.task
        isaac_args.headless = args.headless
        isaac_args.num_envs = 1  # Only need 1 environment for data collection
        
        # Initialize Isaac Gym environment
        print("Initializing Isaac Gym environment...")
        env, env_cfg = task_registry.make_env(name=args.task, args=isaac_args)
        print("Isaac Gym environment initialized!")
        print()
        
        # Collect trajectories from simulation
        print("Starting data collection...")
        print()
        
        successful = 0
        failed = 0
        
        try:
            for traj_idx in range(args.num_trajectories):
                success = collect_trajectory_isaac_gym(
                    env,
                    traj_idx,
                    args.output_dir,
                    traj_duration=args.traj_duration,
                    frequency=args.frequency
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                # Print progress
                if (traj_idx + 1) % 10 == 0 or traj_idx == args.num_trajectories - 1:
                    print(f"Progress: {traj_idx + 1}/{args.num_trajectories} "
                          f"({successful} successful, {failed} failed)")
                    print()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        # Save dataset metadata
        dataset_metadata = {
            'num_trajectories': successful,
            'traj_duration': args.traj_duration,
            'frequency': args.frequency,
            'collection_timestamp': datetime.now().isoformat(),
            'source': 'isaac_gym',
            'task': args.task,
        }
    
    metadata_filename = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(metadata_filename, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print()
    print("=" * 60)
    print("Data collection complete!")
    print(f"  Successful trajectories: {successful}")
    print(f"  Failed trajectories: {failed}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
