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

import numpy as np
import os
import signal
import sys
import types
from datetime import datetime

import isaacgym
from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

# Global variable to store the runner for signal handler
ppo_runner_global = None

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) to save checkpoint before exiting."""
    if ppo_runner_global is not None and ppo_runner_global.log_dir is not None:
        print("\n\nInterrupt received! Saving checkpoint before exiting...")
        try:
            checkpoint_path = os.path.join(ppo_runner_global.log_dir, 
                                         f'model_{ppo_runner_global.current_learning_iteration}.pt')
            ppo_runner_global.save(checkpoint_path)
            print(f"Checkpoint saved successfully to: {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    print("Exiting...")
    sys.exit(0)

def train(args, headless=False):
    global ppo_runner_global
    args.headless = headless
    # args.resume = True
    
    # Get environment config and modify it for straight arms and weighted hands
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Set arm positions to be mostly straight and in front for training
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
    
    # Ensure initial position is set correctly to 1.0m (higher to prevent emerging from ground)
    spawn_height = 0.75
    if not hasattr(env_cfg.init_state, 'pos') or len(env_cfg.init_state.pos) < 3:
        env_cfg.init_state.pos = [0.0, 0.0, spawn_height]
    else:
        env_cfg.init_state.pos[2] = spawn_height  # Always set to spawn_height
    
    print(f"Initial spawn position: {env_cfg.init_state.pos}")
    print(f"Terrain mesh type: {env_cfg.terrain.mesh_type}")
    
    # Create environment with modified config
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Set base_init_state height to spawn_height (this is what actually gets used)
    if hasattr(env, 'base_init_state'):
        env.base_init_state[2] = spawn_height
        print(f"Set base_init_state height to: {env.base_init_state[2]}")
    
    # Patch reset function to ensure robots always spawn at correct height
    original_reset_root_states = env._reset_root_states
    def reset_root_states_with_height(self, env_ids):
        """Wrapper that ensures robots spawn at correct height"""
        # Call original reset function
        original_reset_root_states(env_ids)
        # Force z position to be at least spawn_height above env_origins
        if hasattr(self, 'env_origins'):
            # Ensure robots spawn at spawn_height above the terrain origin
            self.root_states[env_ids, 2] = self.env_origins[env_ids, 2] + spawn_height
        else:
            # Fallback: just set to spawn_height
            self.root_states[env_ids, 2] = spawn_height
        # Apply the root states
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    env._reset_root_states = types.MethodType(reset_root_states_with_height, env)
    
    # Fix initial spawn positions (robots may have spawned during environment creation before patching)
    # Manually correct all root states to ensure they're at the correct height
    if hasattr(env, 'root_states') and hasattr(env, 'env_origins'):
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        env.root_states[all_env_ids, 2] = env.env_origins[all_env_ids, 2] + spawn_height
        # Apply the corrected root states
        env_ids_int32 = all_env_ids.to(dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                     gymtorch.unwrap_tensor(env.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        print(f"Fixed initial spawn positions: all robots set to height {spawn_height}m")
    
    # Set arm action scale to 1.0 to allow full range of motion for pose control
    if hasattr(env, 'num_lower_dof') and hasattr(env, 'num_actions'):
        env._arm_action_scale_factor = 1.0  # Full control for our arm poses
        print("Arm movements will be directly controlled with pose sequence")
    else:
        raise ValueError("Could not set arm action scale - required attributes not found")
    
    # Arm movement control setup - same pose system as play.py
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
    
    # Define poses (as target joint angles, converted to action offsets)
    # Actions are offsets from default: action = (target_angle - default_angle) / action_scale
    action_scale = env_cfg.control.action_scale
    
    # Pose 1: Straight against body (arms down, close to body)
    pose_straight_body = {
        'left_shoulder_pitch': (0.0 - left_shoulder_pitch_default) / action_scale,
        'left_shoulder_roll': (0.25 - left_shoulder_roll_default) / action_scale,
        'left_shoulder_yaw': 0.0,
        'left_elbow': ((np.pi/2 - 0.15) - left_elbow_default) / action_scale,
        'right_shoulder_pitch': (0.0 - right_shoulder_pitch_default) / action_scale,
        'right_shoulder_roll': (-0.25 - right_shoulder_roll_default) / action_scale,
        'right_shoulder_yaw': 0.0,
        'right_elbow': ((np.pi/2 - 0.15) - right_elbow_default) / action_scale,
    }
    
    # Base pose for straight front (arms extended forward)
    pose_straight_front_base = {
        'left_shoulder_pitch': (-1.7 - left_shoulder_pitch_default) / action_scale,
        'left_shoulder_roll': (0.25 - left_shoulder_roll_default) / action_scale,
        'left_shoulder_yaw': 0.0,
        'left_elbow': ((np.pi/2 - 0.15) - left_elbow_default) / action_scale,
        'right_shoulder_pitch': (-1.7 - right_shoulder_pitch_default) / action_scale,
        'right_shoulder_roll': (-0.25 - right_shoulder_roll_default) / action_scale,
        'right_shoulder_yaw': 0.0,
        'right_elbow': ((np.pi/2 - 0.15) - right_elbow_default) / action_scale,
    }
    
    # Pose 2a: Straight front with LOW noise (small random variations)
    pose_straight_front_low_noise = pose_straight_front_base.copy()
    
    # Pose 2b: Straight front with HIGH noise (will be randomized continuously)
    pose_straight_front_high_noise = pose_straight_front_base.copy()
    
    # Pose 3: Bent 90 degrees (elbow at 90 degrees) - matching play.py
    pose_bent_90 = {
        'left_shoulder_pitch': (0.0 - left_shoulder_pitch_default) / action_scale,  # Arms down
        'left_shoulder_roll': (0.0 - left_shoulder_roll_default) / action_scale,
        'left_shoulder_yaw': 0.0,
        'left_elbow': (0.0 - left_elbow_default) / action_scale,  # Straight (matching play.py)
        'right_shoulder_pitch': (0.0 - right_shoulder_pitch_default) / action_scale,
        'right_shoulder_roll': (0.0 - right_shoulder_roll_default) / action_scale,
        'right_shoulder_yaw': 0.0,
        'right_elbow': (0.0 - right_elbow_default) / action_scale,  # Straight (matching play.py)
    }
    
    # Pose list: straight_body, straight_front_low_noise, straight_front_high_noise, bent_90
    poses = [pose_straight_body, pose_straight_front_low_noise, pose_straight_front_high_noise, pose_bent_90]
    pose_names = ["straight_body", "straight_front_low_noise", "straight_front_high_noise", "bent_90"]
    
    # Convert poses to tensors for vectorized operations
    # Order: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
    #        right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
    pose_keys = ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
                 'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow']
    poses_tensor = torch.zeros((len(poses), len(pose_keys)), device=env.device)
    for i, pose in enumerate(poses):
        for j, key in enumerate(pose_keys):
            poses_tensor[i, j] = pose[key]
    pose_straight_front_base_tensor = poses_tensor[1]  # Index 1 is straight_front_low_noise (same base as high noise)
    
    # Noise parameters - use same as play.py
    low_noise_scale = 0.1   # 10% variation for low noise pose
    high_noise_scale = 4  # Same as play.py - 300% variation for high noise pose
    
    # Time distribution (HOMIE curriculum): 40% high noise, 40% low noise, 10% straight body, 10% bent 90
    pose_weights = [0.15, 0.30, 0.40, 0.15]  # Probabilities for each pose
    pose_hold_times = [4.0, 4.0, 4.0, 4.0]  # Base hold times (will be scaled by weights)
    
    # Calculate actual hold times based on weights (normalize to a cycle time)
    cycle_time = 10.0  # Total cycle time in seconds
    total_weight = sum(pose_weights)
    pose_hold_times = [cycle_time * w / total_weight for w in pose_weights]
    pose_hold_times_tensor = torch.tensor(pose_hold_times, device=env.device)
    
    transition_time = 2.0  # Transition between poses over 2 seconds
    high_noise_interp_duration = 1.0  # Duration for each random interpolation (1 second)
    
    # Vectorized pose state using tensors
    initial_pose_indices = torch.tensor(np.random.choice(len(poses), size=env.num_envs, p=pose_weights), 
                                         device=env.device, dtype=torch.long)
    
    # Initialize pose state tensors
    env.pose_state = {
        'current_pose_idx': initial_pose_indices.clone(),  # Which pose each env is currently in (0-3)
        'pose_start_time': torch.zeros(env.num_envs, device=env.device),  # When current pose started (in seconds)
        'in_transition': torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),  # Whether currently transitioning
        'transition_start_time': torch.zeros(env.num_envs, device=env.device),  # When transition started
        'transition_start_pose': poses_tensor[initial_pose_indices].clone(),  # Starting pose for transition [num_envs, 8]
        'transition_target_pose': poses_tensor[initial_pose_indices].clone(),  # Target pose for transition [num_envs, 8]
        'high_noise_start_pose': pose_straight_front_base_tensor.unsqueeze(0).repeat(env.num_envs, 1).clone(),  # [num_envs, 8]
        'high_noise_target_pose': pose_straight_front_base_tensor.unsqueeze(0).repeat(env.num_envs, 1).clone(),  # [num_envs, 8]
        'high_noise_interp_start_time': torch.zeros(env.num_envs, device=env.device),  # When high-noise interpolation started
        'env_step_counter': torch.zeros(env.num_envs, dtype=torch.long, device=env.device),  # Track steps per environment
    }
    
    # Pre-compute pose weights for random selection (cumulative distribution)
    pose_weights_cumsum = torch.tensor(np.cumsum(pose_weights), device=env.device)
    
    # Store pose system parameters in environment
    env.pose_system = {
        'poses_tensor': poses_tensor,  # [4, 8] tensor of all poses
        'pose_straight_front_base_tensor': pose_straight_front_base_tensor,  # [8] base pose tensor
        'pose_keys': pose_keys,  # Order of joints in tensor
        'pose_names': pose_names,
        'pose_weights': pose_weights,
        'pose_weights_cumsum': pose_weights_cumsum,  # Cumulative weights for random selection
        'pose_hold_times_tensor': pose_hold_times_tensor,  # Hold times as tensor
        'low_noise_scale': low_noise_scale,
        'high_noise_scale': high_noise_scale,
        'transition_time': transition_time,
        'high_noise_interp_duration': high_noise_interp_duration,
        'left_shoulder_pitch_idx': left_shoulder_pitch_idx,
        'left_shoulder_roll_idx': left_shoulder_roll_idx,
        'left_shoulder_yaw_idx': left_shoulder_yaw_idx,
        'left_elbow_idx': left_elbow_idx,
        'left_wrist_roll_idx': left_wrist_roll_idx,
        'left_wrist_pitch_idx': left_wrist_pitch_idx,
        'left_wrist_yaw_idx': left_wrist_yaw_idx,
        'right_shoulder_pitch_idx': right_shoulder_pitch_idx,
        'right_shoulder_roll_idx': right_shoulder_roll_idx,
        'right_shoulder_yaw_idx': right_shoulder_yaw_idx,
        'right_elbow_idx': right_elbow_idx,
        'right_wrist_roll_idx': right_wrist_roll_idx,
        'right_wrist_pitch_idx': right_wrist_pitch_idx,
        'right_wrist_yaw_idx': right_wrist_yaw_idx,
    }
    
    # Disable upper body curriculum to prevent it from overwriting our pose actions
    # Store original upper_interval and set it to a very large number so curriculum never updates
    if hasattr(env.cfg.domain_rand, 'upper_interval'):
        env._original_upper_interval = env.cfg.domain_rand.upper_interval
        env.cfg.domain_rand.upper_interval = 999999999  # Effectively disable curriculum updates
    
    # Monkey-patch the step function to apply pose system
    original_step = env.step
    original_reset_idx = env.reset_idx
    
    def reset_idx_with_poses(self, env_ids):
        """Vectorized wrapper around reset_idx() that resets pose state for reset environments"""
        env_obj = self
        # Call original reset_idx function first
        result = original_reset_idx(env_ids)
        
        # Reset pose state for environments that are resetting (AFTER original reset)
        if len(env_ids) > 0:
            state = env_obj.pose_state
            ps = env_obj.pose_system
            
            # Convert env_ids to tensor if needed
            if isinstance(env_ids, torch.Tensor):
                env_ids_tensor = env_ids
            else:
                env_ids_tensor = torch.tensor(env_ids, device=env_obj.device, dtype=torch.long)
            
            num_reset = len(env_ids_tensor)
            
            # Vectorized pose selection using weighted random
            rand_vals = torch.rand(num_reset, device=env_obj.device)
            next_pose_indices = torch.searchsorted(ps['pose_weights_cumsum'], rand_vals)
            next_pose_indices = torch.clamp(next_pose_indices, 0, len(ps['pose_names']) - 1)
            
            # Reset pose state (vectorized)
            state['current_pose_idx'][env_ids_tensor] = next_pose_indices
            state['pose_start_time'][env_ids_tensor] = 0.0
            state['in_transition'][env_ids_tensor] = False
            state['transition_start_time'][env_ids_tensor] = 0.0
            state['transition_start_pose'][env_ids_tensor] = ps['poses_tensor'][next_pose_indices]
            state['transition_target_pose'][env_ids_tensor] = ps['poses_tensor'][next_pose_indices]
            state['high_noise_start_pose'][env_ids_tensor] = ps['pose_straight_front_base_tensor'].unsqueeze(0).repeat(num_reset, 1)
            state['high_noise_target_pose'][env_ids_tensor] = ps['pose_straight_front_base_tensor'].unsqueeze(0).repeat(num_reset, 1)
            state['high_noise_interp_start_time'][env_ids_tensor] = 0.0
            state['env_step_counter'][env_ids_tensor] = 0
            
            # Calculate pose actions for reset environments (vectorized)
            num_upper_dof = env_obj.num_actions - env_obj.num_lower_dof
            reset_arm_actions = torch.zeros((env_obj.num_envs, num_upper_dof), device=env_obj.device)
            
            # Get current poses for reset environments
            current_poses = ps['poses_tensor'][next_pose_indices]  # [num_reset, 8]
            
            # Add wrist randomization (vectorized)
            wrist_noise = torch.randn(num_reset, 6, device=env_obj.device) * ps['high_noise_scale']
            
            # Set arm actions (vectorized)
            reset_arm_actions[env_ids_tensor, ps['left_shoulder_pitch_idx']] = current_poses[:, 0]
            reset_arm_actions[env_ids_tensor, ps['left_shoulder_roll_idx']] = current_poses[:, 1]
            reset_arm_actions[env_ids_tensor, ps['left_shoulder_yaw_idx']] = current_poses[:, 2]
            reset_arm_actions[env_ids_tensor, ps['left_elbow_idx']] = current_poses[:, 3]
            reset_arm_actions[env_ids_tensor, ps['left_wrist_roll_idx']] = wrist_noise[:, 0]
            reset_arm_actions[env_ids_tensor, ps['left_wrist_pitch_idx']] = wrist_noise[:, 1]
            reset_arm_actions[env_ids_tensor, ps['left_wrist_yaw_idx']] = wrist_noise[:, 2]
            
            reset_arm_actions[env_ids_tensor, ps['right_shoulder_pitch_idx']] = current_poses[:, 4]
            reset_arm_actions[env_ids_tensor, ps['right_shoulder_roll_idx']] = current_poses[:, 5]
            reset_arm_actions[env_ids_tensor, ps['right_shoulder_yaw_idx']] = current_poses[:, 6]
            reset_arm_actions[env_ids_tensor, ps['right_elbow_idx']] = current_poses[:, 7]
            reset_arm_actions[env_ids_tensor, ps['right_wrist_roll_idx']] = wrist_noise[:, 3]
            reset_arm_actions[env_ids_tensor, ps['right_wrist_pitch_idx']] = wrist_noise[:, 4]
            reset_arm_actions[env_ids_tensor, ps['right_wrist_yaw_idx']] = wrist_noise[:, 5]
            
            # Restore pose actions after reset (the original reset_idx zeroed them)
            env_obj.current_upper_actions[env_ids_tensor] = reset_arm_actions[env_ids_tensor]
            env_obj.delta_upper_actions[env_ids_tensor] = torch.zeros_like(reset_arm_actions[env_ids_tensor])
        
        return result
    
    def step_with_poses(self, actions):
        """Vectorized wrapper around step() that applies pose system to upper body actions"""
        env_obj = self
        ps = env_obj.pose_system
        state = env_obj.pose_state
        
        # Increment step counters (vectorized)
        state['env_step_counter'] += 1
        env_times = state['env_step_counter'].float() * env_obj.dt
        
        num_upper_dof = env_obj.num_actions - env_obj.num_lower_dof
        num_arm_joints = 8  # 4 joints per arm (shoulder pitch/roll/yaw, elbow)
        
        # Get current hold times for each environment (vectorized)
        current_hold_times = ps['pose_hold_times_tensor'][state['current_pose_idx']]
        time_in_pose = env_times - state['pose_start_time']
        
        # Check which environments need to start transitions
        should_start_transition = (~state['in_transition']) & (time_in_pose >= current_hold_times)
        
        if should_start_transition.any():
            # Select next poses (vectorized weighted random selection)
            rand_vals = torch.rand(should_start_transition.sum(), device=env_obj.device)
            # Use searchsorted for efficient weighted random selection
            next_pose_indices = torch.searchsorted(ps['pose_weights_cumsum'], rand_vals)
            next_pose_indices = torch.clamp(next_pose_indices, 0, len(ps['pose_names']) - 1)
            
            # Update state for environments starting transitions
            state['in_transition'][should_start_transition] = True
            state['transition_start_time'][should_start_transition] = env_times[should_start_transition]
            state['transition_start_pose'][should_start_transition] = ps['poses_tensor'][state['current_pose_idx'][should_start_transition]]
            
            # Set new pose indices
            old_indices = state['current_pose_idx'][should_start_transition].clone()
            state['current_pose_idx'][should_start_transition] = next_pose_indices
            
            # Handle high noise pose initialization
            entering_high_noise = (next_pose_indices == 2)  # straight_front_high_noise
            if entering_high_noise.any():
                # Create mask for environments entering high noise
                entering_mask = torch.zeros(env_obj.num_envs, dtype=torch.bool, device=env_obj.device)
                entering_mask[should_start_transition] = entering_high_noise
                
                state['high_noise_start_pose'][entering_mask] = state['transition_start_pose'][entering_mask].clone()
                state['high_noise_target_pose'][entering_mask] = ps['pose_straight_front_base_tensor'].unsqueeze(0).repeat(entering_mask.sum(), 1) + \
                    torch.randn(entering_mask.sum(), num_arm_joints, device=env_obj.device) * ps['high_noise_scale']
                state['high_noise_interp_start_time'][entering_mask] = env_times[entering_mask]
                state['transition_target_pose'][entering_mask] = state['high_noise_start_pose'][entering_mask].clone()
            
            # Set transition targets for non-high-noise poses
            not_high_noise_mask = should_start_transition.clone()
            not_high_noise_mask[should_start_transition] = ~entering_high_noise
            if not_high_noise_mask.any():
                state['transition_target_pose'][not_high_noise_mask] = ps['poses_tensor'][state['current_pose_idx'][not_high_noise_mask]]
        
        # Check which transitions are complete
        transition_progress = (env_times - state['transition_start_time']) / ps['transition_time']
        transition_complete = state['in_transition'] & (transition_progress >= 1.0)
        
        if transition_complete.any():
            state['in_transition'][transition_complete] = False
            state['pose_start_time'][transition_complete] = env_times[transition_complete]
            transition_progress[transition_complete] = 1.0
            
            # Initialize high noise for environments entering high noise pose
            entering_high_noise_complete = transition_complete & (state['current_pose_idx'] == 2)
            if entering_high_noise_complete.any():
                state['high_noise_start_pose'][entering_high_noise_complete] = state['transition_target_pose'][entering_high_noise_complete].clone()
                state['high_noise_target_pose'][entering_high_noise_complete] = ps['pose_straight_front_base_tensor'].unsqueeze(0).repeat(entering_high_noise_complete.sum(), 1) + \
                    torch.randn(entering_high_noise_complete.sum(), num_arm_joints, device=env_obj.device) * ps['high_noise_scale']
                state['high_noise_interp_start_time'][entering_high_noise_complete] = env_times[entering_high_noise_complete]
        
        # Compute current poses (vectorized)
        # Start with base poses for each environment
        current_poses = ps['poses_tensor'][state['current_pose_idx']].clone()  # [num_envs, 8]
        
        # Handle transitions: interpolate between start and target
        in_transition_mask = state['in_transition']
        if in_transition_mask.any():
            smooth_progress = transition_progress[in_transition_mask]
            smooth_progress = smooth_progress * smooth_progress * (3.0 - 2.0 * smooth_progress)  # Ease in-out
            current_poses[in_transition_mask] = state['transition_start_pose'][in_transition_mask] + \
                smooth_progress.unsqueeze(1) * (state['transition_target_pose'][in_transition_mask] - state['transition_start_pose'][in_transition_mask])
        
        # Handle high noise pose: linear interpolation
        high_noise_mask = (state['current_pose_idx'] == 2) & (~state['in_transition'])
        if high_noise_mask.any():
            time_since_interp = env_times[high_noise_mask] - state['high_noise_interp_start_time'][high_noise_mask]
            interp_progress = time_since_interp / ps['high_noise_interp_duration']
            
            # Generate new targets for completed interpolations
            need_new_target = interp_progress >= 1.0
            if need_new_target.any():
                state['high_noise_start_pose'][high_noise_mask][need_new_target] = state['high_noise_target_pose'][high_noise_mask][need_new_target].clone()
                state['high_noise_target_pose'][high_noise_mask][need_new_target] = ps['pose_straight_front_base_tensor'].unsqueeze(0).repeat(need_new_target.sum(), 1) + \
                    torch.randn(need_new_target.sum(), num_arm_joints, device=env_obj.device) * ps['high_noise_scale']
                state['high_noise_interp_start_time'][high_noise_mask][need_new_target] = env_times[high_noise_mask][need_new_target]
                interp_progress[need_new_target] = 0.0
            
            # Linear interpolation
            current_poses[high_noise_mask] = state['high_noise_start_pose'][high_noise_mask] + \
                interp_progress.unsqueeze(1) * (state['high_noise_target_pose'][high_noise_mask] - state['high_noise_start_pose'][high_noise_mask])
        
        # Add noise to low noise and other poses
        low_noise_mask = (state['current_pose_idx'] == 1) & (~state['in_transition'])
        other_poses_mask = (~state['in_transition']) & (state['current_pose_idx'] != 2) & (state['current_pose_idx'] != 1)
        noise_mask = low_noise_mask | other_poses_mask
        if noise_mask.any():
            noise = torch.randn(noise_mask.sum(), num_arm_joints, device=env_obj.device) * ps['low_noise_scale']
            current_poses[noise_mask] += noise
        
        # Build arm actions tensor [num_envs, num_upper_dof]
        arm_actions = torch.zeros((env_obj.num_envs, num_upper_dof), device=env_obj.device)
        
        # Set arm joint actions (vectorized)
        arm_actions[:, ps['left_shoulder_pitch_idx']] = current_poses[:, 0]
        arm_actions[:, ps['left_shoulder_roll_idx']] = current_poses[:, 1]
        arm_actions[:, ps['left_shoulder_yaw_idx']] = current_poses[:, 2]
        arm_actions[:, ps['left_elbow_idx']] = current_poses[:, 3]
        arm_actions[:, ps['right_shoulder_pitch_idx']] = current_poses[:, 4]
        arm_actions[:, ps['right_shoulder_roll_idx']] = current_poses[:, 5]
        arm_actions[:, ps['right_shoulder_yaw_idx']] = current_poses[:, 6]
        arm_actions[:, ps['right_elbow_idx']] = current_poses[:, 7]
        
        # Add wrist randomization (vectorized)
        wrist_noise = torch.randn(env_obj.num_envs, 6, device=env_obj.device) * ps['high_noise_scale']
        arm_actions[:, ps['left_wrist_roll_idx']] = wrist_noise[:, 0]
        arm_actions[:, ps['left_wrist_pitch_idx']] = wrist_noise[:, 1]
        arm_actions[:, ps['left_wrist_yaw_idx']] = wrist_noise[:, 2]
        arm_actions[:, ps['right_wrist_roll_idx']] = wrist_noise[:, 3]
        arm_actions[:, ps['right_wrist_pitch_idx']] = wrist_noise[:, 4]
        arm_actions[:, ps['right_wrist_yaw_idx']] = wrist_noise[:, 5]
        
        # Override the environment's upper body actions BEFORE step
        env_obj.current_upper_actions = arm_actions
        env_obj.delta_upper_actions = torch.zeros_like(arm_actions)
        
        # Call original step function
        return original_step(actions)
    
    print(f"Arm control initialized with HOMIE curriculum")
    print(f"  - Pose distribution: {[f'{w*100:.0f}%' for w in pose_weights]}")
    print(f"  - Hold times: {[f'{t:.1f}s' for t in pose_hold_times]}")
    print(f"  - Low noise scale: {low_noise_scale*100}%")
    print(f"  - High noise scale: {high_noise_scale*100}%")
    print(f"  - Transition time: {transition_time}s")
    print(f"  - Wrist joints: free-moving (uncontrolled)")
    
    # Verify initial robot positions after setup
    print(f"Number of environments: {env.num_envs}")
    print(f"Base init state z-position: {env.base_init_state[2]}")
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner_global = ppo_runner
    
    # Patch the environment's step function using MethodType to properly bind it
    # This ensures the function is properly bound as an instance method
    env.step = types.MethodType(step_with_poses, env)
    env.reset_idx = types.MethodType(reset_idx_with_poses, env)
    
    # Also patch the runner's env reference (should be same object, but patch both to be safe)
    ppo_runner.env.step = types.MethodType(step_with_poses, ppo_runner.env)
    ppo_runner.env.reset_idx = types.MethodType(reset_idx_with_poses, ppo_runner.env)
    
    # Final fix: Ensure all robots are at correct height right before training starts
    # This handles any edge cases where positions might have been reset during initialization
    if hasattr(env, 'root_states') and hasattr(env, 'env_origins'):
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        env.root_states[all_env_ids, 2] = env.env_origins[all_env_ids, 2] + spawn_height
        env_ids_int32 = all_env_ids.to(dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                     gymtorch.unwrap_tensor(env.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        print(f"Final check: All robots at height {spawn_height}m before training starts")
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        # This will be handled by the signal handler, but we catch it here as a fallback
        signal_handler(signal.SIGINT, None)

if __name__ == '__main__':
    args = get_args()
    train(args, headless=args.headless)
