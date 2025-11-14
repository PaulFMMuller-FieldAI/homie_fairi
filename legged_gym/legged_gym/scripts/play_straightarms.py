# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys

import onnxruntime as ort

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.xr_teleop_interface import XRTeleopInterface

import numpy as np
import torch
import argparse


def load_policy():
    body = torch.jit.load("", map_location="cuda:0")
    def policy(obs):
        action = body.forward(obs)
        return action
    return policy

def load_onnx_policy():
    model = ort.InferenceSession("")
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference

def play_straightarms(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74, use_xr_teleop=False, xr_websocket_url="ws://localhost:8080", enable_pedals=True, checkpoint_path=None):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Force only 1 environment for play mode (override args.num_envs if set)
    args.num_envs = 1
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.asset.self_collision = 0
    
    # Set arm positions to be mostly straight and in front for play mode
    # Shoulder pitch: positive angle to raise arms forward
    # Shoulder roll: small negative angle to spread arms slightly outward
    # Elbow: small angle to keep mostly straight
    env_cfg.init_state.default_joint_angles["left_shoulder_pitch_joint"] = -1.7  # Raise arm forward
    env_cfg.init_state.default_joint_angles["left_shoulder_roll_joint"] = 0.25  # Spread arm outward
    env_cfg.init_state.default_joint_angles["left_shoulder_yaw_joint"] = 0.0
    env_cfg.init_state.default_joint_angles["left_elbow_joint"] = 3.14/2 - 0.15  # Slightly bent, mostly straight
    
    env_cfg.init_state.default_joint_angles["right_shoulder_pitch_joint"] = -1.7  # Raise arm forward
    env_cfg.init_state.default_joint_angles["right_shoulder_roll_joint"] = -0.25  # Spread arm outward (opposite direction)
    env_cfg.init_state.default_joint_angles["right_shoulder_yaw_joint"] = 0.0
    env_cfg.init_state.default_joint_angles["right_elbow_joint"] = 3.14/2 - 0.15  # Slightly bent, mostly straight
    
    # Initialize XR teleoperation interface if enabled, or if pedals are enabled
    xr_interface = None
    if use_xr_teleop or enable_pedals:
        # Enable upper body teleoperation only if XR is enabled
        if use_xr_teleop:
            env_cfg.env.upper_teleop = True
        else:
            env_cfg.env.upper_teleop = False
        
        xr_interface = XRTeleopInterface(
            xr_websocket_url=xr_websocket_url,
            enable_pedals=enable_pedals,
            enable_xr=use_xr_teleop,  # Only enable WebSocket if XR teleop is requested
            command_scale={
                'x_vel': env_cfg.commands.ranges.lin_vel_x[1],
                'y_vel': env_cfg.commands.ranges.lin_vel_y[1],
                'yaw_vel': env_cfg.commands.ranges.ang_vel_yaw[1],
                'height': 0.5,  # Max height adjustment
            },
            smoothing_factor=0.8,
            deadzone=0.05
        )
        xr_interface.start()
        if use_xr_teleop:
            print("XR Teleoperation enabled")
            print(f"  - WebSocket: {xr_websocket_url}")
        if enable_pedals:
            print("Pedal control enabled")
            print(f"  - Pedals: Enabled")
    else:
        env_cfg.env.upper_teleop = False
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Add weight to hands for play mode
    # Set hand payload mass (in kg) - this adds weight to the hands
    hand_weight = 0.6  # 600g per hand
    if hasattr(env, 'hand_payload') and hasattr(env, 'left_hand_index') and hasattr(env, 'right_hand_index'):
        env.hand_payload[:, 0] = hand_weight  # Left hand
        env.hand_payload[:, 1] = hand_weight  # Right hand
        # Apply the hand payload to the rigid body properties
        for i in range(env.num_envs):
            props = env.gym.get_actor_rigid_body_properties(env.envs[i], env.actor_handles[i])
            props[env.left_hand_index].mass = env.default_rigid_body_mass[env.left_hand_index] + env.hand_payload[i, 0]
            props[env.right_hand_index].mass = env.default_rigid_body_mass[env.right_hand_index] + env.hand_payload[i, 1]
            env.gym.set_actor_rigid_body_properties(env.envs[i], env.actor_handles[i], props)
        print(f"Set hand weights to {hand_weight} kg per hand")
    else:
        print("Warning: Could not set hand weights - required attributes not found")
        raise ValueError("Could not set hand weights - required attributes not found")
    
    # We'll directly control arm movements, so no need to scale them down
    # Set arm action scale to 1.0 to allow full range of motion
    if hasattr(env, 'num_lower_dof') and hasattr(env, 'num_actions'):
        env._arm_action_scale_factor = 1.0  # Full control for our arm poses
        print("Arm movements will be directly controlled with pose sequence")
    else:
        raise ValueError("Could not set arm action scale - required attributes not found")
    
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    env.commands[:, 4] = height
    env.action_curriculum_ratio = 1.0
    obs = env.get_observations()
    
    # load policy
    # If checkpoint_path is provided, use it; otherwise use resume from config
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        # Set resume to False if we're loading a specific checkpoint
        train_cfg.runner.resume = False
    else:
        train_cfg.runner.resume = True
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    # Load checkpoint if path is provided (after runner is created)
    if checkpoint_path is not None:
        ppo_runner.load(checkpoint_path)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
        
        # Compile and save JIT policy next to checkpoint
        try:
            checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            jit_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_jit.pt")
            
            # Get the actor_critic model for compilation
            actor_critic = ppo_runner.alg.actor_critic
            
            # Compile with torch.jit based on model type
            if hasattr(actor_critic, 'estimator'):
                # HIM model with estimator
                from legged_gym.utils.helpers import PolicyExporterHIM
                exporter = PolicyExporterHIM(actor_critic)
                exporter.to('cpu')
                jit_model = torch.jit.script(exporter)
            else:
                # Standard actor model
                model = actor_critic.actor.to('cpu')
                jit_model = torch.jit.script(model)
            
            # Save compiled model
            jit_model.save(jit_path)
            print(f"Compiled and saved JIT policy to: {jit_path}")
        except Exception as e:
            print(f"Warning: Failed to compile JIT policy: {e}")
    
    policy = ppo_runner.get_inference_policy(device=env.device) # Use this to load from trained pt file
    
    # policy = load_onnx_policy() # Use this to load from exported onnx file
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    print(policy)
    
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    
    # Arm movement control setup
    # Get joint indices for arm joints
    dof_names = env.dof_names
    left_shoulder_pitch_idx = dof_names.index("left_shoulder_pitch_joint") - env.num_lower_dof
    left_shoulder_roll_idx = dof_names.index("left_shoulder_roll_joint") - env.num_lower_dof
    left_shoulder_yaw_idx = dof_names.index("left_shoulder_yaw_joint") - env.num_lower_dof
    left_elbow_idx = dof_names.index("left_elbow_joint") - env.num_lower_dof
    left_wrist_roll_idx = dof_names.index("left_wrist_roll_joint") - env.num_lower_dof
    left_wrist_pitch_idx = dof_names.index("left_wrist_pitch_joint") - env.num_lower_dof
    left_wrist_yaw_idx = dof_names.index("left_wrist_yaw_joint") - env.num_lower_dof
    
    right_shoulder_pitch_idx = dof_names.index("right_shoulder_pitch_joint") - env.num_lower_dof
    right_shoulder_roll_idx = dof_names.index("right_shoulder_roll_joint") - env.num_lower_dof
    right_shoulder_yaw_idx = dof_names.index("right_shoulder_yaw_joint") - env.num_lower_dof
    right_elbow_idx = dof_names.index("right_elbow_joint") - env.num_lower_dof
    right_wrist_roll_idx = dof_names.index("right_wrist_roll_joint") - env.num_lower_dof
    right_wrist_pitch_idx = dof_names.index("right_wrist_pitch_joint") - env.num_lower_dof
    right_wrist_yaw_idx = dof_names.index("right_wrist_yaw_joint") - env.num_lower_dof
    
    # Get default joint positions for arms to compute offsets
    default_dof_pos = env.default_dof_pos[0].cpu().numpy()  # Get default positions
    left_shoulder_pitch_default = default_dof_pos[dof_names.index("left_shoulder_pitch_joint")]
    left_shoulder_roll_default = default_dof_pos[dof_names.index("left_shoulder_roll_joint")]
    left_elbow_default = default_dof_pos[dof_names.index("left_elbow_joint")]
    right_shoulder_pitch_default = default_dof_pos[dof_names.index("right_shoulder_pitch_joint")]
    right_shoulder_roll_default = default_dof_pos[dof_names.index("right_shoulder_roll_joint")]
    right_elbow_default = default_dof_pos[dof_names.index("right_elbow_joint")]
    
    # Define three poses (as target joint angles, converted to action offsets)
    # Actions are offsets from default: action = (target_angle - default_angle) / action_scale
    action_scale = env_cfg.control.action_scale
    
    # Pose 1: Straight against body (arms down, close to body)
    pose_straight_body = {
        'left_shoulder_pitch': (0.0 - left_shoulder_pitch_default) / action_scale,  # Raise arm forward
        'left_shoulder_roll': (0.25 - left_shoulder_roll_default) / action_scale,  # Spread slightly outward
        'left_shoulder_yaw': 0.0,
        'left_elbow': ((np.pi/2 - 0.15) - left_elbow_default) / action_scale,  # Mostly straight
        'right_shoulder_pitch': (0.0 - right_shoulder_pitch_default) / action_scale,
        'right_shoulder_roll': (-0.25 - right_shoulder_roll_default) / action_scale,  # Opposite direction
        'right_shoulder_yaw': 0.0,
        'right_elbow': ((np.pi/2 - 0.15) - right_elbow_default) / action_scale,
    }
    
    # Base pose for straight front (arms extended forward)
    pose_straight_front_base = {
        'left_shoulder_pitch': (-1.7 - left_shoulder_pitch_default) / action_scale,  # Raise arm forward
        'left_shoulder_roll': (0.25 - left_shoulder_roll_default) / action_scale,  # Spread slightly outward
        'left_shoulder_yaw': 0.0,
        'left_elbow': ((np.pi/2 - 0.15) - left_elbow_default) / action_scale,  # Mostly straight
        'right_shoulder_pitch': (-1.7 - right_shoulder_pitch_default) / action_scale,
        'right_shoulder_roll': (-0.25 - right_shoulder_roll_default) / action_scale,  # Opposite direction
        'right_shoulder_yaw': 0.0,
        'right_elbow': ((np.pi/2 - 0.15) - right_elbow_default) / action_scale,
    }
    
    # Pose 2a: Straight front with LOW noise (small random variations)
    pose_straight_front_low_noise = pose_straight_front_base.copy()
    
    # Pose 2b: Straight front with HIGH noise (will be randomized continuously)
    # This pose will be regenerated with high noise each time we enter it
    pose_straight_front_high_noise = pose_straight_front_base.copy()
    
    # Pose 3: Bent 90 degrees (elbow at 90 degrees)
    pose_bent_90 = {
        'left_shoulder_pitch': (0.0 - left_shoulder_pitch_default) / action_scale,  # Arms down
        'left_shoulder_roll': (0.0 - left_shoulder_roll_default) / action_scale,
        'left_shoulder_yaw': 0.0,
        'left_elbow': (np.pi/2 - left_elbow_default) / action_scale,  # 90 degrees bend
        'right_shoulder_pitch': (0.0 - right_shoulder_pitch_default) / action_scale,
        'right_shoulder_roll': (0.0 - right_shoulder_roll_default) / action_scale,
        'right_shoulder_yaw': 0.0,
        'right_elbow': (np.pi/2 - right_elbow_default) / action_scale,  # 90 degrees bend
    }
    
    # Pose list: straight_body, straight_front_low_noise, straight_front_high_noise, bent_90
    poses = [pose_straight_body, pose_straight_front_low_noise, pose_straight_front_high_noise, pose_bent_90]
    pose_names = ["straight_body", "straight_front_low_noise", "straight_front_high_noise", "bent_90"]
    
    # Noise parameters
    low_noise_scale = 0.1   # 10% variation for low noise pose
    high_noise_scale = 2  # 1000% variation for high noise pose
    
    # Time distribution (HOMIE curriculum): 40% high noise, 40% low noise, 10% straight body, 10% bent 90
    pose_weights = [0.1, 0.4, 0.4, 0.1]  # Probabilities for each pose
    pose_hold_times = [3.0, 3.0, 3.0, 3.0]  # Base hold times (will be scaled by weights)
    
    # Calculate actual hold times based on weights (normalize to a cycle time)
    cycle_time = 10.0  # Total cycle time in seconds
    total_weight = sum(pose_weights)
    pose_hold_times = [cycle_time * w / total_weight for w in pose_weights]
    
    # State machine parameters
    transition_time = 2.0  # Transition between poses over 2 seconds
    current_pose_idx = 0
    pose_start_time = 0.0
    in_transition = False
    transition_start_time = 0.0
    transition_start_pose = pose_straight_body.copy()
    transition_target_pose = pose_straight_front_low_noise.copy()
    
    # Initialize current_pose with the starting pose
    current_pose = poses[current_pose_idx].copy()
    
    # High noise pose interpolation state (for HOMIE-style linear interpolation)
    high_noise_current_pose = pose_straight_front_base.copy()  # Current interpolated pose
    high_noise_target_pose = pose_straight_front_base  # Target random pose
    high_noise_start_pose = pose_straight_front_base.copy()  # Start of current interpolation
    high_noise_interp_start_time = 0.0  # When current interpolation started
    high_noise_interp_duration = 1.0  # Duration for each random interpolation (1 second)
    
    # Function to select next pose based on weights
    def select_next_pose():
        """Select next pose based on weighted probabilities"""
        r = np.random.random()
        cumsum = np.cumsum(pose_weights)
        for i, threshold in enumerate(cumsum):
            if r <= threshold:
                return i
        return len(poses) - 1  # Fallback to last pose
    
    # Function to add noise to a pose
    def add_noise_to_pose(pose, noise_scale):
        """Add random noise to a pose"""
        noisy_pose = pose.copy()
        for key in noisy_pose.keys():
            noise = np.random.uniform(-noise_scale, noise_scale)
            noisy_pose[key] += noise
        return noisy_pose
    
    print(f"Arm control initialized with HOMIE curriculum")
    print(f"  - Pose distribution: {[f'{w*100:.0f}%' for w in pose_weights]}")
    print(f"  - Hold times: {[f'{t:.1f}s' for t in pose_hold_times]}")
    print(f"  - Low noise scale: {low_noise_scale*100}%")
    print(f"  - High noise scale: {high_noise_scale*100}%")
    print(f"  - Transition time: {transition_time}s")
    print(f"  - Wrist joints: free-moving (uncontrolled)")
    print(f"Starting with pose: {pose_names[current_pose_idx]}")
    
    # Main control loop
    iteration = 0
    try:
        for _ in range(10*int(env.max_episode_length)):
            env.action_curriculum_ratio = 1.0
            
            # Get actions from policy
            actions = policy(obs.detach())
            
            # Update arm pose state machine
            current_time = iteration * env.dt
            
            # Check if we should transition to next pose
            if in_transition:
                # In transition - interpolate between poses
                transition_progress = (current_time - transition_start_time) / transition_time
                if transition_progress >= 1.0:
                    # Transition complete, hold new pose
                    in_transition = False
                    pose_start_time = current_time
                    transition_progress = 1.0
                    
                    # If entering high noise pose, initialize interpolation state
                    if current_pose_idx == 2:  # straight_front_high_noise
                        high_noise_current_pose = transition_target_pose.copy()
                        high_noise_start_pose = transition_target_pose.copy()
                        high_noise_target_pose = add_noise_to_pose(pose_straight_front_base, high_noise_scale)
                        high_noise_interp_start_time = current_time
                    
                    print(f"Holding pose: {pose_names[current_pose_idx]} (hold time: {pose_hold_times[current_pose_idx]:.1f}s)")
                
                # Smooth interpolation (ease in-out)
                smooth_progress = transition_progress * transition_progress * (3.0 - 2.0 * transition_progress)
                
                # Interpolate each joint
                current_pose = {}
                for key in transition_start_pose.keys():
                    current_pose[key] = transition_start_pose[key] + smooth_progress * (transition_target_pose[key] - transition_start_pose[key])
            else:
                # Not in transition - check if we should start one
                time_in_pose = current_time - pose_start_time
                current_hold_time = pose_hold_times[current_pose_idx]
                
                if time_in_pose >= current_hold_time:
                    # Start transition to next pose (weighted random selection)
                    in_transition = True
                    transition_start_time = current_time
                    transition_start_pose = poses[current_pose_idx].copy()
                    
                    # Select next pose based on weights
                    current_pose_idx = select_next_pose()
                    
                    # Prepare target pose
                    if current_pose_idx == 2:  # straight_front_high_noise
                        # Initialize high noise interpolation state
                        # Start from the base pose (where we're transitioning from)
                        high_noise_current_pose = transition_start_pose.copy()
                        high_noise_start_pose = transition_start_pose.copy()
                        high_noise_target_pose = add_noise_to_pose(pose_straight_front_base, high_noise_scale)
                        high_noise_interp_start_time = current_time
                        # Use the start pose as transition target (will immediately start interpolating)
                        transition_target_pose = high_noise_start_pose.copy()
                    else:
                        transition_target_pose = poses[current_pose_idx].copy()
                    print(f"Transitioning to pose: {pose_names[current_pose_idx]} (will hold for {pose_hold_times[current_pose_idx]:.1f}s)")
                    # Use starting pose for this frame
                    current_pose = transition_start_pose.copy()
                else:
                    # Holding current pose
                    if current_pose_idx == 2:  # High noise pose - HOMIE-style linear interpolation
                        # Check if we've reached the current target
                        time_since_interp_start = current_time - high_noise_interp_start_time
                        interp_progress = time_since_interp_start / high_noise_interp_duration
                        
                        if interp_progress >= 1.0:
                            # Reached target, generate new random target and start new interpolation
                            high_noise_start_pose = high_noise_target_pose.copy()
                            high_noise_target_pose = add_noise_to_pose(pose_straight_front_base, high_noise_scale)
                            high_noise_interp_start_time = current_time
                            interp_progress = 0.0
                        
                        # Linear interpolation between start and target
                        current_pose = {}
                        for key in high_noise_start_pose.keys():
                            current_pose[key] = high_noise_start_pose[key] + interp_progress * (
                                high_noise_target_pose[key] - high_noise_start_pose[key]
                            )
                        high_noise_current_pose = current_pose.copy()
                    elif current_pose_idx == 1:  # Low noise pose - add small variations
                        current_pose = poses[current_pose_idx].copy()
                        current_pose = add_noise_to_pose(current_pose, low_noise_scale)
                    else:  # Other poses - small variations
                        current_pose = poses[current_pose_idx].copy()
                        current_pose = add_noise_to_pose(current_pose, low_noise_scale)
            
            # Override upper body actions with our arm poses
            # Get the current upper body actions tensor
            num_upper_dof = env.num_actions - env.num_lower_dof
            arm_actions = torch.zeros((env.num_envs, num_upper_dof), device=env.device)
            
            # Set arm joint targets (as actions, which will be scaled by action_scale)
            # Left arm
            arm_actions[:, left_shoulder_pitch_idx] = current_pose['left_shoulder_pitch']
            arm_actions[:, left_shoulder_roll_idx] = current_pose['left_shoulder_roll']
            arm_actions[:, left_shoulder_yaw_idx] = current_pose['left_shoulder_yaw']
            arm_actions[:, left_elbow_idx] = current_pose['left_elbow']
            # Wrist joints are free (set to 0)
            arm_actions[:, left_wrist_roll_idx] = 0.0
            arm_actions[:, left_wrist_pitch_idx] = 0.0
            arm_actions[:, left_wrist_yaw_idx] = 0.0
            
            # Right arm
            arm_actions[:, right_shoulder_pitch_idx] = current_pose['right_shoulder_pitch']
            arm_actions[:, right_shoulder_roll_idx] = current_pose['right_shoulder_roll']
            arm_actions[:, right_shoulder_yaw_idx] = current_pose['right_shoulder_yaw']
            arm_actions[:, right_elbow_idx] = current_pose['right_elbow']
            # Wrist joints are free (set to 0)
            arm_actions[:, right_wrist_roll_idx] = 0.0
            arm_actions[:, right_wrist_pitch_idx] = 0.0
            arm_actions[:, right_wrist_yaw_idx] = 0.0
            
            # Override the environment's upper body actions and prevent curriculum from modifying them
            env.current_upper_actions = arm_actions
            env.delta_upper_actions = torch.zeros_like(arm_actions)  # Prevent curriculum from modifying
            
            # Get teleoperation commands if enabled (XR or pedals)
            if xr_interface:
                commands = xr_interface.get_commands()
                env.commands[:, 0] = commands['x_vel']
                env.commands[:, 1] = commands['y_vel']
                env.commands[:, 2] = commands['yaw_vel']
                env.commands[:, 4] = commands['height']
                
                # Handle upper body teleoperation if enabled (XR only)
                if use_xr_teleop and env_cfg.env.upper_teleop and xr_interface.get_upper_body_enabled():
                    left_hand, right_hand = xr_interface.get_hand_poses()
                    if left_hand is not None and right_hand is not None:
                        # Map hand poses to upper body actions
                        # This is a placeholder - you'll need to implement IK or direct mapping
                        # based on your robot's upper body configuration
                        # For now, we'll use the existing upper body curriculum system
                        pass
            else:
                # Use static commands
                env.commands[:, 0] = x_vel
                env.commands[:, 1] = y_vel
                env.commands[:, 2] = yaw_vel
                env.commands[:, 4] = height
            
            # Step environment
            obs, _, _, _, _, _, _ = env.step(actions.detach())
            
            # Move camera if enabled
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)
            
            iteration += 1
            
            # Print teleoperation status periodically (XR or pedals)
            if xr_interface and enable_pedals:
                commands = xr_interface.get_commands()
                pedal_states = xr_interface.get_pedal_states()
                
                # Print every 10 iterations (more frequent feedback)
                if iteration % 10 == 0:
                    pedal_str = "Pedals: "
                    pedal_str += f"a={1 if pedal_states['left'] else 0} "
                    pedal_str += f"b={1 if pedal_states['forward'] else 0} "
                    pedal_str += f"c={1 if pedal_states['right'] else 0}"
                    print(f"[{iteration:5d}] {pedal_str} | "
                          f"x_vel={commands['x_vel']:6.3f} "
                          f"y_vel={commands['y_vel']:6.3f} "
                          f"yaw_vel={commands['yaw_vel']:6.3f} "
                          f"height={commands['height']:5.3f}m")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        if xr_interface:
            xr_interface.stop()
            print("XR Teleoperation interface stopped")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    
    # Parse additional arguments for XR teleoperation FIRST
    # This must be done before get_args() since get_args() consumes all sys.argv
    parser = argparse.ArgumentParser(description='Additional XR teleoperation arguments', add_help=False)
    parser.add_argument('--use-xr-teleop', action='store_true', 
                        help='Enable XR teleoperation (Apple Vision Pro + pedals)')
    parser.add_argument('--xr-websocket-url', type=str, default='ws://localhost:8080',
                        help='WebSocket URL for Unitree XR teleoperate server')
    parser.add_argument('--enable-pedals', action='store_true', default=True,
                        help='Enable pedal input support')
    parser.add_argument('--disable-pedals', action='store_false', dest='enable_pedals',
                        help='Disable pedal input support')
    parser.add_argument('--x-vel', type=float, default=0.0,
                        help='Static x velocity (ignored if XR teleop enabled)')
    parser.add_argument('--y-vel', type=float, default=0.0,
                        help='Static y velocity (ignored if XR teleop enabled)')
    parser.add_argument('--yaw-vel', type=float, default=0.0,
                        help='Static yaw velocity (ignored if XR teleop enabled)')
    parser.add_argument('--height', type=float, default=0.74,
                        help='Static height (ignored if XR teleop enabled)')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Direct path to checkpoint file (e.g., /path/to/model_2600.pt). Overrides config resume settings.')
    
    # Parse only known args and remove them from sys.argv before calling get_args()
    xr_args, remaining_argv = parser.parse_known_args()
    
    # Remove XR-specific arguments from sys.argv so get_args() doesn't see them
    xr_arg_names = ['--use-xr-teleop', '--disable-pedals', '--xr-websocket-url', 
                    '--enable-pedals', '--x-vel', '--y-vel', '--yaw-vel', 
                    '--height', '--checkpoint-path']
    filtered_argv = [sys.argv[0]]  # Keep script name
    skip_next = False
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if arg in xr_arg_names:
            # Skip this argument and its value if it takes one
            if arg in ['--xr-websocket-url', '--x-vel', '--y-vel', '--yaw-vel', 
                       '--height', '--checkpoint-path']:
                skip_next = True  # Skip the value too
            continue
        filtered_argv.append(arg)
    
    # Temporarily replace sys.argv, call get_args(), then restore
    original_argv = sys.argv.copy()
    sys.argv = filtered_argv
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    
    play_straightarms(
        args,
        x_vel=xr_args.x_vel,
        y_vel=xr_args.y_vel,
        yaw_vel=xr_args.yaw_vel,
        height=xr_args.height,
        use_xr_teleop=xr_args.use_xr_teleop,
        xr_websocket_url=xr_args.xr_websocket_url,
        enable_pedals=xr_args.enable_pedals,
        checkpoint_path=xr_args.checkpoint_path
    )

