# /IOTCONNECT dashboard reference

Everything the RZ/V2H robotic-arm demo sends to /IOTCONNECT and every command
it accepts, with the exact parameters. Use it to decide what to put on a
dashboard and how to wire each button.

- **Template:** [`robarmv2h-template.json`](robarmv2h-template.json). The
  device must be on this template, or the newer fields and commands below will
  not exist.
- **Telemetry interval:** every 5 seconds.
- **Command parameters:** typed as plain text in the command's parameter box.
  "none" means leave it empty.
- **`<board-ip>`** below is the board's address on your network (it is assigned
  by DHCP and can change after a power cycle).

Not yet exercised from the cloud: `set_backend`, `set_yolo_conf` and
`calibrate auto`.

---

## 1. Live readouts worth showing

| Widget | Attribute | Notes |
|---|---|---|
| Demo state | `state` | SCANNING, TRACKING, GRABBING, HOLDING, and the transport/drop steps |
| Balls moved | `demoStats.drops` | Since the app started |
| Grab success % | `demoStats.grab_success_pct` | Held ÷ attempts |
| Grab attempts | `demoStats.grab_attempts` | Times the gripper closed on a tracked ball |
| Grabs held | `demoStats.grabs_held` | Ball still held after the lift |
| Grabs empty | `demoStats.grabs_empty` | Jaws closed on nothing |
| Grabs slipped | `demoStats.grabs_slipped` | Ball lost during the lift |
| Last cycle time | `demoStats.last_cycle_s` | Seconds between the last two drops |
| Average cycle time | `demoStats.avg_cycle_s` | Seconds between drops |
| App uptime | `demoStats.uptime_min` | Minutes |
| Inference backend | `ballTrack.backend` | `DRP-AI` or `CPU` |
| Inference time | `ballTrack.yolo_ms` | Milliseconds |
| Loop rate | `ballTrack.npu_fps` | Frames per second |
| Ball position | `ballTrack.ball_x`, `ballTrack.ball_y` | Pixels in the 640×480 camera frame |
| Ball size | `ballTrack.ball_r` | Apparent radius in pixels; grows as the gripper closes in |
| Aim error | `ballTrack.pan_err`, `ballTrack.tilt_err` | Pixels from the frame centre |
| Grab gate | `ballTrack.gate_mode`, `ballTrack.r_target` | Current gate and radius target |
| Shoulder pan | `shoulder_pan` | Servo 6, 0–1000; rotates the base |
| Shoulder lift | `shoulder_lift` | Servo 5, 0–1000 |
| Elbow | `elbow_flex` | Servo 4, 0–1000 |
| Wrist pitch | `wrist_flex` | Servo 3, 0–1000 |
| Wrist roll | `wrist_roll` | Servo 2, 0–1000 |
| Gripper | `gripper` | Servo 1, 0–1000; low = open |
| CPU load | `sysInfo_cpu.usage_percent` | % |
| CPU temperature | `sysInfo_cpu.temp` | °C |
| Memory used | `sysInfo_memory.percent` | % |
| Disk used | `sysInfo_storage.percent` | % |
| Board uptime | `sysInfo_system.uptime` | Text |

While the arm is tracking, `gripper` and `wrist_roll` show the last commanded
value rather than a fresh reading; the other four joints are read every frame.

### Leave these off

They are always 0 or unused on this board:

- `sysInfo_gpu.usage_percent`, `sysInfo_gpu.temp`
- `sysInfo_memory.temp`
- `ballTrack.depth_ms`, `ballTrack.depth_at_ball`,
  `ballTrack.depth_at_ball_smoothed`, `ballTrack.depth_settle`,
  `ballTrack.D_grab`, `ballTrack.D_tolerance`, `ballTrack.D_threshold`
  (there is no depth model installed)

### Other fields in the template

Available if you want them, but not needed for a demo dashboard:

- `ballTrack.state`, `is_prediction`, `radius_err`, `velocity_x`, `velocity_y`,
  `pred_frames_left`, `no_ball_frames`, `hold_target`, `d_pan`, `d_tilt`,
  `d_lift`, `d_elbow` — tracker internals.
- `sysInfo_system.hostname`, `cpu_brand`, `cpu_vendor`, `cpu_mhz`,
  `cpu_physical_cores`, `architecture`, `system`, `release`, `platform` —
  static board identification.
- `sysInfo_cpu.top_process_name`, `top_process_cmd`, `top_process_cpu_percent`
  and `sysInfo_memory.total`, `available`, `used`, `top_process_name`,
  `top_process_cmd`, `top_process_mem`, `sysInfo_storage.total`, `used`,
  `free` — system detail.

---

## 2. Commands: run the demo

| Command | Parameter | What it does |
|---|---|---|
| `set_mode` | `yolo-pickplace` | Start picking balls and dropping them in the box |
| `set_mode` | `yolo-ball` | Find and grab a ball, then hold it (no drop) |
| `set_mode` | `idle` | Stop moving; stay connected |
| `stop_demo` | none | Same as `set_mode idle` |
| `set_backend` | `drpai`, `cpu` or `auto` | Switch the detector between the DRP-AI accelerator and the CPU. Restarts the running mode. Saved across reboots |
| `set_yolo_conf` | a number 0.05–0.95, or `auto` | Detection confidence; applies live and is saved. `auto` = the model's default (0.30) |
| `yolo_config_show` | none | Replies with the saved backend and confidence, and what is running now |

## 3. Commands: grab tuning

| Command | Parameter | What it does |
|---|---|---|
| `set_grab_radius` | pixels 10–400, or `auto` | How big the ball must look before the gripper closes. Applies live. `auto` = the taught value |
| `set_grab_gate` | `radius`, `depth` or `both` | Which signal triggers the grab. Use `radius`: there is no depth model on this board |
| `grab_threshold_show` | none | Replies with the current gate settings |
| `grab_threshold_reset` | none | Back to defaults. **Loses a tuned radius** |

## 4. Commands: camera

| Command | Parameter | What it does |
|---|---|---|
| `camera_preset` | a preset name | Load a saved picture setup, live. Built in: `auto`, `bright`, `cool`, `dim`, `normal`, `vivid`, `warm`; plus any you saved |
| `camera_setting` | `<name> <value>`, e.g. `exposure 400` | Change one setting, live and saved. Names: `auto_exposure`, `exposure`, `gain`, `auto_wb`, `wb_temperature`, `brightness`, `contrast`, `saturation`, `sharpness`, `gamma`, `hue` |
| `camera_preset_save` | a name (letters, digits, `_`, `-`) | Save the current settings as a preset |
| `camera_settings_show` | none | Replies with the current settings |
| `camera_presets_show` | none | Replies with all presets |
| `camera_settings_reset` | none | Clears the settings. **Loses the tuned picture** |

`auto_exposure` is `1` for manual and `3` for automatic. `exposure` is in
0.1 ms steps on the Innomaker camera.

## 5. Commands: teaching poses and calibration

| Command | Parameter | What it does |
|---|---|---|
| `release_torque` | none | Servos go limp so the arm can be posed by hand. **The arm will drop — support it first** |
| `hold_pose` | none | Re-engage the servos where the arm is now |
| `teach_scan_pose` | `center`, `left` or `right` | Save the current pose as that search pose. Used from the next mode start |
| `scan_poses_show` | none | Replies with the saved search poses |
| `scan_poses_reset` | none | Deletes them. **Falls back to built-in poses that may not suit your rig** |
| `teach_drop_pose` | none | Save the current pose as the drop position |
| `drop_pose_show` | none | Replies with the saved drop pose |
| `drop_pose_reset` | none | Deletes it. **Pick-and-place then has no drop target** |
| `calibrate` | `auto` | Hands-off calibration: the arm moves by itself. Progress at `http://<board-ip>:8000/` |
| `stop_calibration` | none | Stop a running calibrator |

To teach a pose: `release_torque`, pose the arm by hand, `hold_pose`, then
`teach_scan_pose <name>` or `teach_drop_pose`.

## 6. Commands: manual jog and scripted moves

Use these with the demo in `idle`. While it is picking, the tracker overrides
them.

| Command | Parameter | What it does |
|---|---|---|
| `move_left` | none | Base (servo 6) +100 |
| `move_right` | none | Base (servo 6) −100 |
| `move_forward` | none | Shoulder lift (servo 5) −100 |
| `move_backward` | none | Shoulder lift (servo 5) +100 |
| `move_up` | none | Elbow (servo 4) −100 |
| `move_down` | none | Elbow (servo 4) +100 |
| `wrist_flex_up` | none | Wrist pitch (servo 3) −100 |
| `wrist_flex_down` | none | Wrist pitch (servo 3) +100 |
| `wrist_roll_cw` | none | Wrist roll (servo 2) +100 |
| `wrist_roll_ccw` | none | Wrist roll (servo 2) −100 |
| `open_gripper` | none | Gripper (servo 1) −100 |
| `close_gripper` | none | Gripper (servo 1) +100 |
| `move_to` | six comma-separated positions, 0–1000 each | Move all six joints. Order: gripper, wrist roll, wrist flex, elbow, shoulder lift, base. Example: `130,543,865,291,505,400` |
| `move_to_home` | none | Go to the built-in "home" pose (elbow folded back over the base) |
| `demo_wave` | none | Scripted wave; ends in the home pose |
| `demo_bow` | none | Scripted bow; ends in the home pose |
| `demo_stretch` | none | Scripted stretch upward; ends in the home pose |
| `demo_scan` | none | Scripted side-to-side sweep; ends in the home pose |
| `demo_shake_no` | none | Scripted "no" shake; ends in the home pose |
| `demo_pickup` | none | Scripted pick-and-place motion without vision; ends in the home pose |

---

## 7. In the template but not useful on this board

These belong to the colour-tracking and depth modes, which need calibration
files this demo does not ship with. Leave them off the dashboard:

- `set_mode ball`, `set_mode pickplace`, and the shortcut commands
  `set_mode_ball`, `set_mode_pickplace`
- `calibrate ball`, `calibrate box`, `calibrate offset`,
  `calibrate grab_depth`, and the shortcut commands `calibrate_ball`,
  `calibrate_box`, `calibrate_offset`, `calibrate_grab_depth`
- `set_grab_threshold` (the depth floor of the grab gate)

`set_mode_yolo_ball` and `set_mode_yolo_pickplace` work, but duplicate
`set_mode` with a parameter.

## 8. Live video

The camera view is not a dashboard item. It is shown full screen on the
board's HDMI output and served at `http://<board-ip>:8080` on the local
network.
