# XR Teleoperation Setup for HomieRL

This guide explains how to use Apple Vision Pro and pedals for teleoperating your robot using Unitree's XR teleoperate system.

## Overview

The XR teleoperation system integrates:
- **Apple Vision Pro**: Head tracking for locomotion commands (forward/backward, lateral, turning)
- **Pedals**: Additional input devices for braking and control
- **Unitree XR Teleoperate**: WebSocket-based communication protocol

## Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install websocket-client pygame
   ```

2. **Unitree XR Teleoperate Server**: 
   - Make sure you have Unitree's XR teleoperate server running
   - Default WebSocket URL: `ws://localhost:8080`
   - Refer to Unitree's documentation for setup instructions

3. **Apple Vision Pro Setup**:
   - Ensure your Apple Vision Pro is connected to the same network
   - The XR teleoperate server should be receiving tracking data from the Vision Pro

4. **Pedal Setup**:
   - Connect your pedals to the host PC
   - They should appear as a joystick/gamepad device
   - The system will auto-detect pedals on startup

## Usage

### Basic Usage with XR Teleoperation

```bash
python legged_gym/legged_gym/scripts/play.py --task g1 --use-xr-teleop
```

### Custom WebSocket URL

If your Unitree XR server is running on a different address:

```bash
python legged_gym/legged_gym/scripts/play.py --task g1 --use-xr-teleop --xr-websocket-url ws://192.168.1.100:8080
```

### Disable Pedals

If you only want to use Vision Pro (no pedals):

```bash
python legged_gym/legged_gym/scripts/play.py --task g1 --use-xr-teleop --disable-pedals
```

### Without XR Teleoperation (Static Commands)

```bash
python legged_gym/legged_gym/scripts/play.py --task g1 --x-vel 0.5 --y-vel 0.0 --yaw-vel 0.1 --height 0.74
```

## Control Mapping

### Apple Vision Pro Head Tracking

- **Forward/Backward Tilt**: Controls forward/backward velocity (x_vel)
  - Tilt head forward → Move forward
  - Tilt head backward → Move backward

- **Left/Right Tilt**: Controls lateral velocity (y_vel)
  - Tilt head right → Move right
  - Tilt head left → Move left

- **Head Rotation**: Controls yaw velocity (turning)
  - Rotate head left/right → Turn robot

- **Hand Height**: Controls robot height
  - Raise hand above head → Increase height
  - Lower hand below head → Decrease height

### Pedals

- **Left Pedal**: Can be used for additional control (customizable)
- **Right Pedal**: Can be used for additional control (customizable)
- **Brake Pedal**: Reduces all velocities by 90% (emergency stop)

## Configuration

### Command Scaling

You can adjust the command scaling in the `play()` function call. The default scales are:

```python
command_scale = {
    'x_vel': 1.2,      # m/s max forward/backward
    'y_vel': 0.5,      # m/s max lateral
    'yaw_vel': 0.8,    # rad/s max turning
    'height': 0.5      # m max height adjustment
}
```

### Smoothing and Deadzone

The interface includes built-in smoothing and deadzone filtering:
- **Smoothing Factor**: 0.8 (higher = smoother, more lag)
- **Deadzone**: 0.05 (commands below this threshold are ignored)

## Data Format

The XR teleoperation interface expects data from Unitree's XR server in the following JSON format:

```json
{
    "head": {
        "position": [x, y, z],
        "rotation": [qx, qy, qz, qw],
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
```

## Troubleshooting

### WebSocket Connection Issues

- Check that the Unitree XR server is running
- Verify the WebSocket URL is correct
- Ensure your network allows WebSocket connections
- Check firewall settings

### Pedal Not Detected

- Verify pedals are connected via USB
- Check they appear as a joystick/gamepad device
- Try running `python -m pygame.examples.joystick` to test detection
- The system will auto-detect pedals, but you may need to adjust button/axis mappings in the code

### No Commands Received

- Check that Apple Vision Pro is sending data to the XR server
- Verify the WebSocket connection is established (check console output)
- Ensure the data format matches the expected JSON structure

### Commands Too Sensitive/Not Sensitive Enough

- Adjust `command_scale` in the `play()` function
- Modify `smoothing_factor` (higher = smoother, less responsive)
- Adjust `deadzone` (higher = need more movement to register)

## Upper Body Teleoperation

Upper body teleoperation is enabled when both hands are tracked. Currently, this is a placeholder for future implementation. You'll need to:

1. Implement inverse kinematics (IK) for mapping hand poses to joint angles
2. Or implement direct pose mapping based on your robot's configuration
3. Update the upper body action handling in `play.py` (line 148-156)

## Integration with Unitree XR Teleoperate

This interface is designed to work with Unitree's XR teleoperate system. Make sure:

1. The Unitree XR server is running and accessible
2. The WebSocket endpoint matches the configured URL
3. The data format matches what the interface expects (see Data Format section)

For more information about Unitree's XR teleoperate system, refer to their official documentation and GitHub repository.

## Safety Notes

- Always test in a safe environment first
- The brake pedal provides emergency stopping capability
- Monitor the robot's behavior during teleoperation
- Have an emergency stop mechanism ready
- Start with low command scales and increase gradually

## Example Workflow

1. Start Unitree XR teleoperate server
2. Put on Apple Vision Pro and ensure tracking is active
3. Connect pedals to the host PC
4. Run the play script with `--use-xr-teleop` flag
5. Verify WebSocket connection in console output
6. Start moving your head to control the robot
7. Use pedals for additional control/braking

## Future Improvements

- [ ] Implement full upper body IK for hand tracking
- [ ] Add more sophisticated command filtering
- [ ] Support for custom pedal mappings
- [ ] Real-time command visualization
- [ ] Recording and playback of teleoperation sessions

