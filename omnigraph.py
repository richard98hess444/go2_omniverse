# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
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


import omni
import omni.graph.core as og


def create_front_cam_omnigraph(robot_num):
    """Define the OmniGraph for the Isaac Sim environment."""

    keys = og.Controller.Keys

    graph_path = f"/ROS_" + f"front_cam{robot_num}"
    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("IsaacCreateRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("ROS2CameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],

            keys.SET_VALUES: [
                ("IsaacCreateRenderProduct.inputs:cameraPrim", f"/World/envs/env_{robot_num}/Robot/base/front_cam"),
                ("IsaacCreateRenderProduct.inputs:enabled", True),
                ("ROS2CameraHelper.inputs:type", "rgb"),
                ("ROS2CameraHelper.inputs:topicName", f"robot{robot_num}/front_cam/rgb"),
                ("ROS2CameraHelper.inputs:frameId", f"robot{robot_num}"),
            ],

            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "IsaacCreateRenderProduct.inputs:execIn"),
                ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelper.inputs:execIn"),
                ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelper.inputs:renderProductPath"),
            ],

        },
    )

# def create_ros2_clock():
#     graph_path = f"/ROS_Clock"
#     og.Controller.edit(
#         {"graph_path": graph_path, 
#          "evaluator_name": "execution",
#          "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,},
#         {
#             og.Controller.Keys.CREATE_NODES: [
#                 ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
#                 ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
#                 ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
#                 # ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
#                 ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
#             ],
#             og.Controller.Keys.CONNECT: [
#                 # Connecting execution of OnImpulseEvent node to PublishClock so it will only publish when an impulse event is triggered
#                 # ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
#                 ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
#                 # Connecting simulationTime data of ReadSimTime to the clock publisher node
#                 ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
#                 # Connecting the ROS2 Context to the clock publisher node so it will run under the specified ROS2 Domain ID
#                 ("Context.outputs:context", "PublishClock.inputs:context"),
#             ],
#             og.Controller.Keys.SET_VALUES: [
#                 # Assigning topic name to clock publisher
#                 ("PublishClock.inputs:topicName", "/clock"),
#                 # Assigning a Domain ID of 1 to Context node
#                 ("Context.inputs:domain_id", 1),
#             ],
#         },
#     )