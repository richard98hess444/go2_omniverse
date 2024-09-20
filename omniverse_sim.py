"""Script to play a checkpoint if an RL agent from RSL-RL."""
from __future__ import annotations


"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.orbit.app import AppLauncher


import cli_args  
import time
import os
import threading


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--custom_env", type=str, default="office", help="Setup the environment")
parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
parser.add_argument("--terrain", type=str, default="rough", help="Setup the robot")
parser.add_argument("--robot_amount", type=int, default=1, help="Setup the robot amount")


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni


ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

# FOR VR SUPPORT
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.core", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.steamvr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.simulatedxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.openxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.telemetry", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.profile.vr", True)


"""Rest everything follows."""
import gymnasium as gym
import torch
import carb


from omni.isaac.orbit_tasks.utils import get_checkpoint_path
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper
)
import omni.isaac.orbit.sim as sim_utils
import omni.appwindow
from rsl_rl.runners import OnPolicyRunner



import rclpy
from ros2 import RobotBaseNode, add_camera, add_rtx_lidar, pub_robo_data_ros2
from geometry_msgs.msg import Twist


from agent_cfg import unitree_go2_agent_cfg, unitree_go1_agent_cfg, unitree_g1_agent_cfg, unitree_go1_flat_agent_cfg
from custom_rl_env import UnitreeGo2CustomEnvCfg, UnitreeGo1RoughEnvCfg, G1RoughEnvCfg, UnitreeGo1FlatEnvCfg
import custom_rl_env

from omnigraph import create_front_cam_omnigraph
global env
import numpy as np


def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                custom_rl_env.base_command["0"] = [1, 0, 0]
            if event.input.name == 'S':
                custom_rl_env.base_command["0"] = [-1, 0, 0]
            if event.input.name == 'A':
                custom_rl_env.base_command["0"] = [0, 1, 0]
            if event.input.name == 'D':
                custom_rl_env.base_command["0"] = [0, -1, 0]
            if event.input.name == 'Q':
                custom_rl_env.base_command["0"] = [0, 0, 1]
            if event.input.name == 'E':
                custom_rl_env.base_command["0"] = [0, 0, -1]
            if event.input.name == "R":
                env.reset()

            if len(custom_rl_env.base_command) > 1:
                if event.input.name == 'I':
                    custom_rl_env.base_command["1"] = [1, 0, 0]
                if event.input.name == 'K':
                    custom_rl_env.base_command["1"] = [-1, 0, 0]
                if event.input.name == 'J':
                    custom_rl_env.base_command["1"] = [0, 1, 0]
                if event.input.name == 'L':
                    custom_rl_env.base_command["1"] = [0, -1, 0]
                if event.input.name == 'U':
                    custom_rl_env.base_command["1"] = [0, 0, 1]
                if event.input.name == 'O':
                    custom_rl_env.base_command["1"] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env.base_command)):
                custom_rl_env.base_command[str(i)] = [0, 0, 0]
    return True

dead_zone = 0.01
pos_sensitivity = 1.0
rot_sensitivity = 1.0
base_command_raw = np.zeros((2, 3))
_INPUT_STICK_VALUE_MAPPING = {
    # forward command
    carb.input.GamepadInput.LEFT_STICK_UP: (0, 0, pos_sensitivity),
    # backward command
    carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 0, pos_sensitivity),
    # right command
    carb.input.GamepadInput.LEFT_STICK_RIGHT: (1, 1, pos_sensitivity),
    # left command
    carb.input.GamepadInput.LEFT_STICK_LEFT: (0, 1, pos_sensitivity),
    # yaw command (positive)
    carb.input.GamepadInput.RIGHT_STICK_RIGHT: (1, 2, rot_sensitivity),
    # yaw command (negative)
    carb.input.GamepadInput.RIGHT_STICK_LEFT: (0, 2, rot_sensitivity),
}

count = 0

def sub_gamepad_event(event, *args, **kwargs) -> bool:
    """https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html"""
    if len(custom_rl_env.base_command) == 0:
        return False

    # check if the event is a button press
    cur_val = event.value
    if abs(cur_val) < dead_zone:
        cur_val = 0

    # button
    if event.input == carb.input.GamepadInput.X:
        if cur_val > 0.5:
            print("[X] pressed")
        else:
            print("[X] released")
    elif event.input == carb.input.GamepadInput.MENU2:
        if cur_val > 0.5:
            env.reset()
            print("Gamepad [START]: Resetting the environment.")

    # left and right stick
    global count
    if event.input in _INPUT_STICK_VALUE_MAPPING:
        direction, axis, value = _INPUT_STICK_VALUE_MAPPING[event.input]
        # change the value only if the stick is moved (soft press)

        base_command_raw[direction, axis] = value * cur_val
        command_sign = base_command_raw[1, :] > base_command_raw[0, :]
        command = base_command_raw.max(axis=0)
        command[command_sign] *= -1
        custom_rl_env.base_command[0] = command
        # count += 1
        # print('##########' + str(count) + '##########')
        # print(direction, axis, value)
        # print('=======================')
        # if direction == 0 and axis == 0:
        #     custom_rl_env.base_command["0"] = [1, 0, 0]
        # if direction == 1 and axis == 0:
        #     custom_rl_env.base_command["0"] = [-1, 0, 0]
        # if direction == 1 and axis == 1:
        #     custom_rl_env.base_command["0"] = [0, 1, 0]
        # if direction == 0 and axis == 1:
        #     custom_rl_env.base_command["0"] = [0, -1, 0]

        # custom_rl_env.base_command["0"] = [1, 0, 0]
    return True


def setup_custom_env():
    try:
        if (args_cli.custom_env == "warehouse" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/warehouse.usd")
            cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))

        if (args_cli.custom_env == "office" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/office.usd")
            cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))
    except:
        print("Error loading custom environment. You should download custom envs folder from: https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing")


def cmd_vel_cb(msg: Twist, num_robot):
    x = msg.linear.x
    y = msg.linear.y
    z = msg.angular.z
    custom_rl_env.base_command[str(num_robot)] = [x, y, z]



def add_cmd_sub(num_envs):
    node_test = rclpy.create_node('position_velocity_publisher')
    for i in range(num_envs):
        node_test.create_subscription(Twist, f'robot{i}/cmd_vel', lambda msg, i=i: cmd_vel_cb(msg, str(i)), 10)
    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node_test,), daemon=True)
    thread.start()



def specify_cmd_for_robots(numv_envs):
    for i in range(numv_envs):
        custom_rl_env.base_command[str(i)] = [0, 0, 0]
def run_sim():
    global env

    # acquire input interface
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    # turn off simulator gamepad camera control (default is enabled)
    # carb_settings_iface = carb.settings.get_settings()
    # carb_settings_iface.set_bool(
    #     "/persistent/app/omniverse/gamepadCameraControl", False
    # )

    # acquire gamepad for controlling the robot
    # _gamepad = _appwindow.get_gamepad(0)
    # _sub_gamepad = _input.subscribe_to_gamepad_events(_gamepad, sub_gamepad_event)

    """Play with RSL-RL agent."""
    # parse configuration
    
    if args_cli.robot == "go2":
        env_cfg = UnitreeGo2CustomEnvCfg()

    if args_cli.robot == "go1":
        env_cfg = UnitreeGo1RoughEnvCfg()

    if args_cli.robot == "go1_flat":
        env_cfg = UnitreeGo1FlatEnvCfg()
    
    if args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()

    # add N robots to env 
    env_cfg.scene.num_envs = args_cli.robot_amount

    # create ros2 camera stream omnigraph
    for i in range(env_cfg.scene.num_envs):
        # create_front_cam_omnigraph(i)
        pass
    specify_cmd_for_robots(env_cfg.scene.num_envs)

    if args_cli.robot == "go2":
        agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg

    if args_cli.robot == "go1":
        agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go1_agent_cfg

    if args_cli.robot == "go1_flat":
        agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go1_flat_agent_cfg

    if args_cli.robot == "g1":
        agent_cfg: RslRlOnPolicyRunnerCfg = unitree_g1_agent_cfg

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"])
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()

    # initialize ROS2 node
    rclpy.init()
    base_node = RobotBaseNode(env_cfg.scene.num_envs)
    add_cmd_sub(env_cfg.scene.num_envs)

    #### Sensors
    # annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot, False)
    # add_camera(env_cfg.scene.num_envs, args_cli.robot)
    # setup_custom_env()
    
    start_time = time.time()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            pub_robo_data_ros2(args_cli.robot, env_cfg.scene.num_envs, base_node, env, None, start_time)
    env.close()