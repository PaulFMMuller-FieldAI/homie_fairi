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

def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74, use_xr_teleop=False, xr_websocket_url="ws://localhost:8080", enable_pedals=True, checkpoint_path=None):

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
    
    # Initialize XR teleoperation interface if enabled
    xr_interface = None
    if use_xr_teleop:
        # Enable upper body teleoperation if XR is enabled
        env_cfg.env.upper_teleop = True
        xr_interface = XRTeleopInterface(
            xr_websocket_url=xr_websocket_url,
            enable_pedals=enable_pedals,
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
        print("XR Teleoperation enabled")
        print(f"  - WebSocket: {xr_websocket_url}")
        print(f"  - Pedals: {'Enabled' if enable_pedals else 'Disabled'}")
    else:
        env_cfg.env.upper_teleop = False
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    env.commands[:, 4] = height
    env.action_curriculum_ratio = 1.0
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    # Load checkpoint if path is provided
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        ppo_runner.load(checkpoint_path)
    
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
    
    # Main control loop
    iteration = 0
    try:
        for _ in range(10*int(env.max_episode_length)):
            env.action_curriculum_ratio = 1.0
            
            # Get actions from policy
            actions = policy(obs.detach())
            
            # Get teleoperation commands if enabled
            if use_xr_teleop and xr_interface:
                commands = xr_interface.get_commands()
                env.commands[:, 0] = commands['x_vel']
                env.commands[:, 1] = commands['y_vel']
                env.commands[:, 2] = commands['yaw_vel']
                env.commands[:, 4] = commands['height']
                
                # Handle upper body teleoperation if enabled
                if env_cfg.env.upper_teleop and xr_interface.get_upper_body_enabled():
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
            
            # Print teleoperation status periodically
            if use_xr_teleop and xr_interface:
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
                          f"height={commands['height']:5.3f}")
    
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
    
    # Parse additional arguments for XR teleoperation first
    # This is done before get_args() to avoid conflicts
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
    
    # Parse XR args first, keeping unknown args for get_args()
    xr_args, remaining_args = parser.parse_known_args()
    
    # Temporarily replace sys.argv to pass only remaining args to get_args()
    original_argv = sys.argv.copy()
    # Filter out the XR-specific arguments from sys.argv
    filtered_argv = [sys.argv[0]]  # Keep script name
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ['--use-xr-teleop', '--disable-pedals']:
            continue
        elif arg in ['--xr-websocket-url', '--x-vel', '--y-vel', '--yaw-vel', '--height', '--checkpoint-path', '--enable-pedals']:
            skip_next = True  # Skip the value too
            continue
        filtered_argv.append(arg)
    
    sys.argv = filtered_argv
    try:
        # Get standard args with filtered arguments
        args = get_args()
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    play(
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