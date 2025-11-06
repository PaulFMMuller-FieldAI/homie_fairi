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
#
# XR Teleoperation Interface for Unitree XR Teleoperate System
# Integrates Apple Vision Pro and pedals for robot control

import json
import threading
import time
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: websocket-client not installed. Install with: pip install websocket-client")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Pedal support may be limited. Install with: pip install pygame")

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not installed. Global keyboard capture not available. Install with: pip install pynput")


class XRTeleopInterface:
    """
    Interface for XR teleoperation using Unitree's XR teleoperate system.
    Integrates Apple Vision Pro head/hand tracking and pedal inputs.
    """
    
    def __init__(
        self,
        xr_websocket_url: str = "ws://localhost:8080",  # Default Unitree XR server
        enable_pedals: bool = True,
        enable_xr: bool = False,  # Enable XR WebSocket connection
        command_scale: Dict[str, float] = None,
        smoothing_factor: float = 0.8,
        deadzone: float = 0.05
    ):
        """
        Initialize XR teleoperation interface.
        
        Args:
            xr_websocket_url: WebSocket URL for Unitree XR teleoperate server
            enable_pedals: Enable pedal input support
            enable_xr: Enable XR WebSocket connection (set to False if only using pedals)
            command_scale: Scaling factors for commands (x_vel, y_vel, yaw_vel, height)
            smoothing_factor: Smoothing factor for command filtering (0-1, higher = smoother)
            deadzone: Deadzone threshold for commands
        """
        self.xr_websocket_url = xr_websocket_url
        self.enable_pedals = enable_pedals
        self.enable_xr = enable_xr
        self.smoothing_factor = smoothing_factor
        self.deadzone = deadzone
        
        # Default command scales (can be overridden)
        self.command_scale = command_scale or {
            'x_vel': 1.2,      # m/s
            'y_vel': 0.5,      # m/s
            'yaw_vel': 0.8,    # rad/s
            'height': 0.5      # m (relative to default)
        }
        
        # Height control parameters
        self.height_step = 0.02  # Height adjustment step per key press (m)
        self.height_min = 0.24   # Minimum height (m)
        self.height_max = 0.74   # Maximum height (robot base height, m)
        
        # Current commands (filtered)
        self.current_commands = {
            'x_vel': 0.0,
            'y_vel': 0.0,
            'yaw_vel': 0.0,
            'height': 0.74,  # Default height
            'upper_body_enabled': False
        }
        
        # Raw commands (before filtering)
        self.raw_commands = {
            'x_vel': 0.0,
            'y_vel': 0.0,
            'yaw_vel': 0.0,
            'height': 0.74,
        }
        
        # XR data from Apple Vision Pro
        self.xr_data = {
            'head_pose': None,      # [x, y, z, qx, qy, qz, qw]
            'left_hand_pose': None, # [x, y, z, qx, qy, qz, qw]
            'right_hand_pose': None,# [x, y, z, qx, qy, qz, qw]
            'head_velocity': None,  # [vx, vy, vz]
            'head_angular_velocity': None,  # [wx, wy, wz]
        }
        
        # Pedal states (keyboard-based: 'a'=left, 'b'=forward, 'c'=right)
        self.pedal_states = {
            'left': False,      # 'a' key
            'forward': False,   # 'b' key
            'right': False,     # 'c' key
            'brake': False,     # Legacy support
        }
        
        # Threading
        self.running = False
        self.ws_thread = None
        self.pedal_thread = None
        self.ws = None
        self.pedal_screen = None  # Keep reference to pygame window
        self.keyboard_listener = None  # pynput keyboard listener
        
        # Initialize pedals if enabled
        # Require pynput for pedal support (global keyboard capture, no window focus needed)
        if self.enable_pedals:
            if not PYNPUT_AVAILABLE:
                raise ImportError(
                    "pynput is required for pedal input support but is not installed. "
                    "Please install it with: pip install pynput\n"
                    "Pedal input requires global keyboard capture which pynput provides."
                )
            print("[PEDAL] Using pynput for global keyboard capture (no window focus needed)")
            print("[PEDAL] Pedal mapping: 'a'=left, 'b'=forward, 'c'=right")
            # Listener will be started in start() method
        
        # Legacy pygame support (deprecated, only if explicitly not using pynput)
        # This code path is kept for backward compatibility but should not be used
        if self.enable_pedals and not PYNPUT_AVAILABLE and PYGAME_AVAILABLE:
            try:
                # Set SDL environment variables for headless operation
                import os
                os.environ['SDL_VIDEODRIVER'] = os.environ.get('SDL_VIDEODRIVER', 'dummy')
                os.environ['SDL_AUDIODRIVER'] = os.environ.get('SDL_AUDIODRIVER', 'dummy')
                
                pygame.init()
                keyboard_available = False
                # Initialize keyboard support (required for key events)
                # Always try to initialize display for keyboard events
                try:
                    pygame.display.init()
                    # Create a small window for keyboard input (required on Linux)
                    # Use a visible window so it can receive focus and keyboard events
                    # Don't use NOFRAME to ensure the window can receive focus properly
                    self.pedal_screen = pygame.display.set_mode((200, 100))
                    # Set window title so it can be identified
                    pygame.display.set_caption("Pedal Input - Keep Focus Here")
                    # Make sure window can receive focus and is visible
                    pygame.display.set_allow_screensaver(True)
                    self.pedal_screen.fill((30, 30, 30))
                    # Draw instructions on the window
                    try:
                        font = pygame.font.Font(None, 24)
                        text = font.render("Pedal Input", True, (255, 255, 255))
                        self.pedal_screen.blit(text, (10, 10))
                        text2 = font.render("a=left, b=forward, c=right", True, (200, 200, 200))
                        self.pedal_screen.blit(text2, (10, 35))
                        pygame.display.flip()
                    except:
                        pygame.display.flip()
                    # Test if keyboard works
                    pygame.key.get_pressed()
                    keyboard_available = True
                    print("[PEDAL] Display initialized successfully. A small window should appear - make sure it has focus to receive keyboard input.")
                    print("[PEDAL] Press 'a' (left), 'b' (forward), or 'c' (right) to control the robot.")
                except Exception as e:
                    print(f"Warning: Could not initialize display for keyboard input: {e}")
                    if not os.environ.get('DISPLAY'):
                        print("  → No DISPLAY environment variable set.")
                        print("  → Consider setting DISPLAY or using X11 forwarding if running over SSH.")
                    keyboard_available = False
                
                pygame.joystick.init()
                self.pedal_joystick = None
                # Try to find a joystick/gamepad (pedals might appear as joystick)
                for i in range(pygame.joystick.get_count()):
                    joystick = pygame.joystick.Joystick(i)
                    joystick.init()
                    name = joystick.get_name()
                    if 'pedal' in name.lower() or 'brake' in name.lower() or pygame.joystick.get_count() == 1:
                        self.pedal_joystick = joystick
                        print(f"Found pedal device: {name}")
                        break
                print("Pedal input initialized (keyboard: a=left, b=forward, c=right)")
                if keyboard_available:
                    print("  ✓ Keyboard input is available")
                else:
                    print("  ✗ Keyboard input is NOT available (no DISPLAY or pygame display not initialized)")
                    print("  → Pedal keyboard input will not work. Consider:")
                    print("    - Setting DISPLAY environment variable")
                    print("    - Using X11 forwarding if running over SSH")
                    print("    - Using a joystick/gamepad instead")
            except Exception as e:
                print(f"Warning: Could not initialize pedals: {e}")
                print("Pedal input will be disabled. Error:", str(e))
                self.enable_pedals = False
        
    def start(self):
        """Start the teleoperation interface."""
        if self.running:
            return
        
        self.running = True
        
        # Start WebSocket connection for XR data (only if XR is enabled)
        if self.enable_xr and WEBSOCKET_AVAILABLE:
            self.ws_thread = threading.Thread(target=self._websocket_loop, daemon=True)
            self.ws_thread.start()
        
        # Start keyboard listener (pynput) - required if pedals are enabled
        if self.enable_pedals:
            if not PYNPUT_AVAILABLE:
                raise ImportError(
                    "pynput is required for pedal input support but is not available. "
                    "Please install it with: pip install pynput"
                )
            try:
                self.keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release
                )
                self.keyboard_listener.start()
                print("[PEDAL] Global keyboard listener started (pynput)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to start pynput keyboard listener: {e}\n"
                    "This is required for pedal input. Please check your system permissions."
                ) from e
        
        # Start pedal polling thread (for pygame or joystick)
        if self.enable_pedals:
            self.pedal_thread = threading.Thread(target=self._pedal_loop, daemon=True)
            self.pedal_thread.start()
        
        print("XR Teleoperation Interface started")
        if self.enable_xr:
            print(f"  - XR WebSocket: {self.xr_websocket_url}")
        print(f"  - Pedals: {'Enabled' if self.enable_pedals else 'Disabled'}")
        if PYNPUT_AVAILABLE and self.enable_pedals:
            print(f"  - Keyboard capture: pynput (global, no focus needed)")
        elif PYGAME_AVAILABLE and self.enable_pedals:
            print(f"  - Keyboard capture: pygame (window focus required)")
    
    def stop(self):
        """Stop the teleoperation interface."""
        self.running = False
        
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        
        # Stop keyboard listener
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
            except:
                pass
        
        if PYGAME_AVAILABLE:
            try:
                pygame.quit()
            except:
                pass
        
        print("XR Teleoperation Interface stopped")
    
    def _websocket_loop(self):
        """WebSocket loop for receiving XR data from Unitree server."""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._process_xr_data(data)
            except Exception as e:
                print(f"Error processing XR data: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
            self.ws = None
        
        def on_open(ws):
            print(f"Connected to XR WebSocket: {self.xr_websocket_url}")
            self.ws = ws
        
        # Connect to WebSocket
        while self.running:
            try:
                ws = websocket.WebSocketApp(
                    self.xr_websocket_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                ws.run_forever()
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                if self.running:
                    time.sleep(2)  # Wait before reconnecting
    
    def _process_xr_data(self, data: Dict):
        """
        Process XR data from Unitree XR teleoperate system.
        
        Expected data format (from Unitree XR system):
        {
            "head": {
                "position": [x, y, z],
                "rotation": [qx, qy, qz, qw],  # quaternion
                "velocity": [vx, vy, vz],
                "angular_velocity": [wx, wy, wz]
            },
            "left_hand": {
                "position": [x, y, z],
                "rotation": [qx, qy, qz, qw]
            },
            "right_hand": {
                "position": [x, y, z],
                "rotation": [qx, qy, qz, qw]
            }
        }
        """
        # Update head pose
        if 'head' in data:
            head = data['head']
            if 'position' in head and 'rotation' in head:
                pos = head['position']
                rot = head['rotation']
                self.xr_data['head_pose'] = pos + rot
                
                # Extract velocity if available
                if 'velocity' in head:
                    self.xr_data['head_velocity'] = head['velocity']
                if 'angular_velocity' in head:
                    self.xr_data['head_angular_velocity'] = head['angular_velocity']
        
        # Update hand poses
        if 'left_hand' in data:
            hand = data['left_hand']
            if 'position' in hand and 'rotation' in hand:
                self.xr_data['left_hand_pose'] = hand['position'] + hand['rotation']
        
        if 'right_hand' in data:
            hand = data['right_hand']
            if 'position' in hand and 'rotation' in hand:
                self.xr_data['right_hand_pose'] = hand['position'] + hand['rotation']
        
        # Convert XR data to locomotion commands
        self._xr_to_commands()
    
    def _xr_to_commands(self):
        """
        Convert Apple Vision Pro head/hand tracking to locomotion commands.
        
        Mapping strategy:
        - Head forward/back tilt -> forward/backward velocity (x_vel)
        - Head left/right tilt -> lateral velocity (y_vel)
        - Head rotation -> yaw velocity (yaw_vel)
        - Hand height -> height command
        """
        if self.xr_data['head_pose'] is None:
            return
        
        # Extract head pose (position + quaternion)
        head_pos = np.array(self.xr_data['head_pose'][:3])
        head_quat = np.array(self.xr_data['head_pose'][3:])  # [qx, qy, qz, qw]
        
        # Convert quaternion to Euler angles for intuitive control
        # Using simplified approach: extract pitch and roll from quaternion
        qx, qy, qz, qw = head_quat
        
        # Convert quaternion to Euler angles (ZYX order)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Map head orientation to velocity commands
        # Forward/backward: pitch (positive = forward tilt = forward velocity)
        pitch_scale = 0.5  # Max pitch angle to consider
        x_vel_raw = -np.clip(pitch / pitch_scale, -1.0, 1.0) * self.command_scale['x_vel']
        
        # Lateral: roll (positive roll = right tilt = right velocity)
        roll_scale = 0.5
        y_vel_raw = -np.clip(roll / roll_scale, -1.0, 1.0) * self.command_scale['y_vel']
        
        # Yaw velocity: use head angular velocity if available, otherwise use yaw angle rate
        if self.xr_data['head_angular_velocity'] is not None:
            yaw_vel_raw = self.xr_data['head_angular_velocity'][2] * self.command_scale['yaw_vel']
        else:
            # Use yaw angle directly (scaled)
            yaw_vel_raw = np.clip(yaw / 1.0, -1.0, 1.0) * self.command_scale['yaw_vel']
        
        # Height: use hand height relative to head
        height_raw = self.current_commands['height']  # Default
        if self.xr_data['left_hand_pose'] is not None:
            hand_pos = np.array(self.xr_data['left_hand_pose'][:3])
            # Height relative to head (positive = hand above head = increase height)
            height_diff = (hand_pos[2] - head_pos[2]) * 0.5  # Scale factor
            height_raw = 0.74 + np.clip(height_diff, -self.command_scale['height'], self.command_scale['height'])
        
        # If pedals are enabled, don't let XR update velocity commands
        # Pedals have exclusive control over velocity when enabled
        # Also, if XR is not enabled, don't process XR data
        if not self.enable_xr:
            return
        
        if self.enable_pedals:
            # Only update height from XR (pedals don't control height)
            # Don't update velocity commands - pedals control those
            if self.xr_data['left_hand_pose'] is not None and head_pos is not None:
                hand_pos = np.array(self.xr_data['left_hand_pose'][:3])
                height_diff = (hand_pos[2] - head_pos[2]) * 0.5
                height_raw = 0.74 + np.clip(height_diff, -self.command_scale['height'], self.command_scale['height'])
                self.raw_commands['height'] = height_raw
                self._filter_commands()
            return
        
        # Original XR command logic (only used when pedals are disabled)
        # Check if pedals are pressed - if so, they override XR commands
        if self.pedal_states['forward'] or self.pedal_states['left'] or self.pedal_states['right']:
            # Pedals override XR head tracking - don't update from XR
            return
        
        # Apply brake pedal modifier if pressed
        if self.pedal_states['brake']:
            # Brake pedal: reduce all velocities
            x_vel_raw *= 0.1
            y_vel_raw *= 0.1
            yaw_vel_raw *= 0.1
        
        # Update raw commands with XR data
        self.raw_commands['x_vel'] = x_vel_raw
        self.raw_commands['y_vel'] = y_vel_raw
        self.raw_commands['yaw_vel'] = yaw_vel_raw
        self.raw_commands['height'] = height_raw
        
        # Apply smoothing and deadzone
        self._filter_commands()
    
    def _pedals_to_commands(self):
        """
        Convert pedal states to velocity commands.
        - 'a' (left): turn left (positive yaw velocity)
        - 'b' (forward): walk forward (positive x velocity)
        - 'c' (right): turn right (negative yaw velocity)
        
        Pedals can be combined: forward + left = forward while turning left
        """
        # Start with zero commands
        x_vel_pedal = 0.0
        y_vel_pedal = 0.0
        yaw_vel_pedal = 0.0
        
        # Forward pedal (b)
        if self.pedal_states['forward']:
            x_vel_pedal = self.command_scale['x_vel']
        
        # Turn left pedal (a) - positive yaw velocity (turn left)
        if self.pedal_states['left']:
            yaw_vel_pedal = self.command_scale['yaw_vel']
        
        # Turn right pedal (c) - negative yaw velocity (turn right)
        # Note: if both left and right are pressed, right takes precedence
        if self.pedal_states['right']:
            yaw_vel_pedal = -self.command_scale['yaw_vel']
        
        # Apply brake if pressed
        if self.pedal_states['brake']:
            x_vel_pedal *= 0.1
            y_vel_pedal *= 0.1
            yaw_vel_pedal *= 0.1
        
        # Update raw commands with pedal input (pedals override XR)
        self.raw_commands['x_vel'] = x_vel_pedal
        self.raw_commands['y_vel'] = y_vel_pedal
        self.raw_commands['yaw_vel'] = yaw_vel_pedal
        # Keep height from current commands (pedals don't control height)
        # self.raw_commands['height'] unchanged
        
        # If no pedals are pressed, immediately set to zero (no smoothing)
        if not (self.pedal_states['forward'] or self.pedal_states['left'] or self.pedal_states['right']):
            # Immediately set to zero when pedals released (no smoothing)
            self.current_commands['x_vel'] = 0.0
            self.current_commands['y_vel'] = 0.0
            self.current_commands['yaw_vel'] = 0.0
        else:
            # Apply smoothing and deadzone only when pedals are active
            self._filter_commands()
    
    def _filter_commands(self):
        """Apply smoothing and deadzone filtering to commands."""
        for key in ['x_vel', 'y_vel', 'yaw_vel']:
            # Smoothing
            self.current_commands[key] = (
                self.smoothing_factor * self.current_commands[key] +
                (1 - self.smoothing_factor) * self.raw_commands[key]
            )
            
            # Deadzone
            if abs(self.current_commands[key]) < self.deadzone:
                self.current_commands[key] = 0.0
        
        # Height doesn't need as much smoothing
        height_alpha = 0.9
        self.current_commands['height'] = (
            height_alpha * self.current_commands['height'] +
            (1 - height_alpha) * self.raw_commands['height']
        )
    
    def _on_key_press(self, key):
        """Callback for pynput keyboard key press events."""
        try:
            # Handle character keys - pynput provides key.char for character keys
            char = None
            if hasattr(key, 'char') and key.char is not None:
                char = key.char  # Keep original case for '+' and '-'
            elif hasattr(key, 'vk') and hasattr(key, 'name'):
                # Try to get character from virtual key code
                # For 'a', 'b', 'c' we can check the name or vk
                if key.name and len(key.name) == 1:
                    char = key.name  # Keep original case
                elif hasattr(key, 'vk'):
                    # Map virtual key codes for a, b, c (65, 66, 67 in ASCII)
                    if key.vk == 65 or key.vk == ord('a'):
                        char = 'a'
                    elif key.vk == 66 or key.vk == ord('b'):
                        char = 'b'
                    elif key.vk == 67 or key.vk == ord('c'):
                        char = 'c'
                    # Map '+' and '-' keys (try common virtual key codes)
                    # On Linux, these might be detected via char, but we try vk as fallback
                    elif key.vk in [0xBB, 0x6B, 0x3D]:  # '+' or '=' key (varies by layout)
                        char = '+' if key.vk != 0x3D else '='
                    elif key.vk in [0xBD, 0x6D]:  # '-' key
                        char = '-'
            
            # Also check if key.name is '=' (which is '+' on US keyboards when shift is pressed)
            # But pynput should give us the actual character, so this is a fallback
            if char is None and hasattr(key, 'name'):
                if key.name == '=':
                    # Check if shift is pressed (on US keyboard, shift+'=' = '+')
                    # For simplicity, we'll treat '=' as '+' for height increase
                    char = '+'
                elif key.name == '-':
                    char = '-'
            
            # Handle pedal keys (lowercase)
            char_lower = char.lower() if char else None
            if char_lower == 'a':
                self.pedal_states['left'] = True
                print("[PEDAL] Key 'a' pressed (left) - pynput")
                self._pedals_to_commands()
            elif char_lower == 'b':
                self.pedal_states['forward'] = True
                print("[PEDAL] Key 'b' pressed (forward) - pynput")
                self._pedals_to_commands()
            elif char_lower == 'c':
                self.pedal_states['right'] = True
                print("[PEDAL] Key 'c' pressed (right) - pynput")
                self._pedals_to_commands()
            # Handle height adjustment keys ('+' or '=' for increase, '-' for decrease)
            elif char == '+' or char == '=':
                # Increase height
                new_height = min(self.current_commands['height'] + self.height_step, self.height_max)
                self.current_commands['height'] = new_height
                self.raw_commands['height'] = new_height
                print(f"[HEIGHT] Increased to {new_height:.3f}m (max: {self.height_max:.3f}m)")
            elif char == '-':
                # Decrease height
                new_height = max(self.current_commands['height'] - self.height_step, self.height_min)
                self.current_commands['height'] = new_height
                self.raw_commands['height'] = new_height
                print(f"[HEIGHT] Decreased to {new_height:.3f}m (min: {self.height_min:.3f}m)")
        except Exception as e:
            print(f"Error in key press handler: {e}")
    
    def _on_key_release(self, key):
        """Callback for pynput keyboard key release events."""
        try:
            # Handle character keys - pynput provides key.char for character keys
            char = None
            if hasattr(key, 'char') and key.char is not None:
                char = key.char.lower()
            elif hasattr(key, 'vk') and hasattr(key, 'name'):
                # Try to get character from virtual key code
                if key.name and len(key.name) == 1:
                    char = key.name.lower()
                elif hasattr(key, 'vk'):
                    # Map virtual key codes for a, b, c
                    if key.vk == 65 or key.vk == ord('a'):
                        char = 'a'
                    elif key.vk == 66 or key.vk == ord('b'):
                        char = 'b'
                    elif key.vk == 67 or key.vk == ord('c'):
                        char = 'c'
            
            if char == 'a':
                self.pedal_states['left'] = False
                print("[PEDAL] Key 'a' released - pynput")
                self._pedals_to_commands()
            elif char == 'b':
                self.pedal_states['forward'] = False
                print("[PEDAL] Key 'b' released - pynput")
                self._pedals_to_commands()
            elif char == 'c':
                self.pedal_states['right'] = False
                print("[PEDAL] Key 'c' released - pynput")
                self._pedals_to_commands()
        except Exception as e:
            print(f"Error in key release handler: {e}")
    
    def _pedal_loop(self):
        """Poll pedal inputs from keyboard (a=left, b=forward, c=right) or joystick."""
        # If pynput is handling keyboard, we don't need pygame at all unless for joystick
        use_pygame_keyboard = not PYNPUT_AVAILABLE
        
        # If pynput is available and handling keyboard, and we don't have pygame initialized,
        # we can skip the entire loop (pynput handles keyboard in callbacks)
        if PYNPUT_AVAILABLE and not PYGAME_AVAILABLE:
            # pynput handles keyboard, no pygame needed
            print("[PEDAL] Using pynput for keyboard, no pygame loop needed")
            # Just sleep to keep thread alive
            import time
            while self.running:
                time.sleep(0.1)
            return
        
        # If no pygame and no pynput, we can't do anything
        if not PYGAME_AVAILABLE:
            return
        
        # Check if pygame was actually initialized (has display)
        pygame_initialized = False
        try:
            pygame_initialized = pygame.display.get_init()
        except:
            pass
        
        # If pygame wasn't initialized and pynput is handling keyboard, skip pygame stuff
        if not pygame_initialized and PYNPUT_AVAILABLE:
            print("[PEDAL] Using pynput for keyboard, pygame not initialized")
            import time
            while self.running:
                time.sleep(0.1)
            return
        
        clock = None
        if pygame_initialized:
            clock = pygame.time.Clock()
        keyboard_available = False
        
        if use_pygame_keyboard:
            try:
                # Check if keyboard is available (display initialized)
                if pygame.display.get_init():
                    pygame.key.get_pressed()  # Test if keyboard works
                    keyboard_available = True
                    print("[PEDAL] Keyboard input is working (pygame)")
                else:
                    print("[PEDAL] Keyboard input NOT available - pygame display not initialized")
            except Exception as e:
                keyboard_available = False
                print(f"[PEDAL] Keyboard input check failed: {e}")
        
        while self.running:
            try:
                # Process pygame events (KEYDOWN/KEYUP) for keyboard input
                # Only if pynput is not handling keyboard and pygame is initialized
                if use_pygame_keyboard and pygame_initialized:
                    try:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                # Handle both key codes and unicode characters
                                if event.key == pygame.K_a or (hasattr(event, 'unicode') and event.unicode == 'a'):
                                    self.pedal_states['left'] = True
                                    print("[PEDAL] Key 'a' pressed (left) - pygame")
                                    self._pedals_to_commands()
                                elif event.key == pygame.K_b or (hasattr(event, 'unicode') and event.unicode == 'b'):
                                    self.pedal_states['forward'] = True
                                    print("[PEDAL] Key 'b' pressed (forward) - pygame")
                                    self._pedals_to_commands()
                                elif event.key == pygame.K_c or (hasattr(event, 'unicode') and event.unicode == 'c'):
                                    self.pedal_states['right'] = True
                                    print("[PEDAL] Key 'c' pressed (right) - pygame")
                                    self._pedals_to_commands()
                            elif event.type == pygame.KEYUP:
                                if event.key == pygame.K_a or (hasattr(event, 'unicode') and event.unicode == 'a'):
                                    self.pedal_states['left'] = False
                                    print("[PEDAL] Key 'a' released - pygame")
                                    self._pedals_to_commands()
                                elif event.key == pygame.K_b or (hasattr(event, 'unicode') and event.unicode == 'b'):
                                    self.pedal_states['forward'] = False
                                    print("[PEDAL] Key 'b' released - pygame")
                                    self._pedals_to_commands()
                                elif event.key == pygame.K_c or (hasattr(event, 'unicode') and event.unicode == 'c'):
                                    self.pedal_states['right'] = False
                                    print("[PEDAL] Key 'c' released - pygame")
                                    self._pedals_to_commands()
                    except Exception as e:
                        # Video system might not be initialized, skip pygame events
                        pass
                    
                    # Also try polling method as backup (for continuous key presses)
                    if keyboard_available:
                        try:
                            keys = pygame.key.get_pressed()
                            # Update pedal states based on current key state
                            self.pedal_states['left'] = keys[pygame.K_a]
                            self.pedal_states['forward'] = keys[pygame.K_b]
                            self.pedal_states['right'] = keys[pygame.K_c]
                            # Update commands if any pedal state changed
                            if keys[pygame.K_a] or keys[pygame.K_b] or keys[pygame.K_c]:
                                self._pedals_to_commands()
                        except Exception as e:
                            # Keyboard not available, skip
                            pass
                elif not pygame_initialized:
                    # pygame not initialized, skip event processing
                    import time
                    time.sleep(0.1)
                    continue
                
                # Also support joystick if available (legacy support)
                # Only if pygame is initialized
                if pygame_initialized and self.pedal_joystick:
                    # Try buttons first
                    num_buttons = self.pedal_joystick.get_numbuttons()
                    if num_buttons >= 3:
                        # Joystick buttons override keyboard if present
                        if self.pedal_joystick.get_button(0):
                            self.pedal_states['left'] = True
                        if self.pedal_joystick.get_button(1):
                            self.pedal_states['forward'] = True
                        if self.pedal_joystick.get_button(2):
                            self.pedal_states['right'] = True
                        self.pedal_states['brake'] = self.pedal_joystick.get_button(3) if num_buttons >= 4 else False
                    
                    # Try axes for analog pedals
                    num_axes = self.pedal_joystick.get_numaxes()
                    if num_axes > 0:
                        # Axis 0 might be left, axis 1 might be forward, axis 2 might be right
                        # Threshold-based detection
                        if num_axes >= 1:
                            axis_val = self.pedal_joystick.get_axis(0)
                            if axis_val > 0.5:
                                self.pedal_states['left'] = True
                        if num_axes >= 2:
                            axis_val = self.pedal_joystick.get_axis(1)
                            if axis_val > 0.5:
                                self.pedal_states['forward'] = True
                        if num_axes >= 3:
                            axis_val = self.pedal_joystick.get_axis(2)
                            if axis_val > 0.5:
                                self.pedal_states['right'] = True
                
                # Update the display to keep the window responsive (only if pygame is initialized)
                if pygame_initialized and self.pedal_screen is not None:
                    try:
                        pygame.display.update()
                    except:
                        pass
                
                # Convert pedal states to velocity commands (only if pedals are pressed)
                # Pedals override XR commands when active
                if self.pedal_states['forward'] or self.pedal_states['left'] or self.pedal_states['right']:
                    self._pedals_to_commands()
                
                # Only use clock.tick if pygame is initialized
                if pygame_initialized:
                    clock.tick(60)  # 60 Hz polling
                else:
                    import time
                    time.sleep(1.0 / 60.0)  # 60 Hz polling without pygame clock
            except Exception as e:
                print(f"Error reading pedals: {e}")
                time.sleep(0.1)
    
    def get_commands(self) -> Dict[str, float]:
        """
        Get current teleoperation commands.
        
        Returns:
            Dictionary with commands: x_vel, y_vel, yaw_vel, height
        """
        return {
            'x_vel': self.current_commands['x_vel'],
            'y_vel': self.current_commands['y_vel'],
            'yaw_vel': self.current_commands['yaw_vel'],
            'height': self.current_commands['height'],
        }
    
    def get_pedal_states(self) -> Dict[str, bool]:
        """Get current pedal states for debugging."""
        return self.pedal_states.copy()
    
    def get_upper_body_enabled(self) -> bool:
        """Check if upper body teleoperation is enabled."""
        # Enable upper body if both hands are tracked
        return (
            self.xr_data['left_hand_pose'] is not None and
            self.xr_data['right_hand_pose'] is not None
        )
    
    def get_hand_poses(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get left and right hand poses."""
        left = np.array(self.xr_data['left_hand_pose'][:7]) if self.xr_data['left_hand_pose'] else None
        right = np.array(self.xr_data['right_hand_pose'][:7]) if self.xr_data['right_hand_pose'] else None
        return left, right


# Example usage and testing
if __name__ == "__main__":
    # Test the interface
    interface = XRTeleopInterface(
        xr_websocket_url="ws://localhost:8080",
        enable_pedals=True
    )
    
    try:
        interface.start()
        print("Interface running. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(0.1)
            commands = interface.get_commands()
            print(f"Commands: x={commands['x_vel']:.2f}, y={commands['y_vel']:.2f}, "
                  f"yaw={commands['yaw_vel']:.2f}, height={commands['height']:.2f}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        interface.stop()

