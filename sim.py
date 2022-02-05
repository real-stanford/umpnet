import math
import os
import pickle
import time

import numpy as np
import open3d as o3d
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

import sim_utils
import utils


class PybulletSim():
    def __init__(self, gui_enabled, action_distance):
        """Pybullet simulation initialization.
        Args:
            gui_enables: bool
            action_distance: float
        """
        if gui_enabled:
            self.bc = bc.BulletClient(connection_mode=pybullet.GUI)
            self.gui = True
        else:
            self.bc = bc.BulletClient(connection_mode=pybullet.DIRECT)
            self.gui = False
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._action_distance = action_distance
        self.ee_urdf = 'assets/suction/suction_with_mount_no_collision.urdf'
        self._mount_force = 1500
        self._mount_speed = 0.002
        self._constraint_force = 2000
        self._mount_base_position = np.array([0, -2, 2])

        self.body_ids = list()
        self.transform_start_to_link = None
        self.normal_to_link = None
        self.link_id = None

        self.toy_joint_vals = {
            'toy1': np.array(pickle.load(open('assets/toy1_joints.pkl', 'rb'))),
            'toy2': np.array(pickle.load(open('assets/toy2_joints.pkl', 'rb')))
        }

        # RGB-D camera setup
        self._scene_cam_image_size = (480, 640)
        self._scene_cam_z_near = 0.01
        self._scene_cam_z_far = 10.0
        self._scene_cam_fov_w = 69.40
        self._scene_cam_focal_length = (float(self._scene_cam_image_size[1])/2)/np.tan((np.pi*self._scene_cam_fov_w/180)/2)
        self._scene_cam_fov_h = (math.atan((float(self._scene_cam_image_size[0])/2)/self._scene_cam_focal_length)*2/np.pi)*180
        self._scene_cam_projection_matrix = self.bc.computeProjectionMatrixFOV(
            fov=self._scene_cam_fov_h,
            aspect=float(self._scene_cam_image_size[1])/float(self._scene_cam_image_size[0]),
            nearVal=self._scene_cam_z_near, farVal=self._scene_cam_z_far
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        self._scene_cam_intrinsics = np.array([[self._scene_cam_focal_length, 0, float(self._scene_cam_image_size[1])/2],
                                             [0, self._scene_cam_focal_length, float(self._scene_cam_image_size[0])/2],
                                             [0, 0, 1]])
    
    
    # Get latest RGB-D image from scene camera
    def get_scene_cam_data(self, cam_position, cam_lookat, cam_up_direction):
        cam_view_matrix = np.array(self.bc.computeViewMatrix(cam_position, cam_lookat, cam_up_direction)).reshape(4, 4).T
        cam_pose_matrix = np.linalg.inv(cam_view_matrix)
        cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]
        camera_data = self.bc.getCameraImage(
            self._scene_cam_image_size[1],
            self._scene_cam_image_size[0],
            self.bc.computeViewMatrix(cam_position, cam_lookat, cam_up_direction),
            self._scene_cam_projection_matrix,
            shadow=1,
            flags=self.bc.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=self.bc.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_pixels = np.array(camera_data[2]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1], 4))
        color_image = rgb_pixels[:,:,:3].astype(np.float32) / 255.0
        z_buffer = np.array(camera_data[3]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1]))
        depth_image = (2.0*self._scene_cam_z_near*self._scene_cam_z_far)/(self._scene_cam_z_far+self._scene_cam_z_near-(2.0*z_buffer-1.0)*(self._scene_cam_z_far-self._scene_cam_z_near))
        segmentation_mask = np.array(camera_data[4], int).reshape(self._scene_cam_image_size)
        
        return color_image, depth_image, segmentation_mask, cam_pose_matrix, cam_view_matrix


    def get_reward(self):
        joint_state = self.joint_states[self.link_id]
        joint_value = self.bc.getJointState(self.body_id, self.link_id)[0]
        reward = abs(joint_value - joint_state['cur_val'])
        sign = joint_state['cur_val'] - joint_state['init_val']
        move_flag = False
        # TODO: heuristic check
        joint_info = self.bc.getJointInfo(self.body_id, self.link_id)
        if joint_info[2] == self.bc.JOINT_REVOLUTE and reward > 0.1:
            move_flag = True
        if joint_info[2] == self.bc.JOINT_PRISMATIC and reward > 0.05:
            move_flag = True
        if reward * 3.5 > (joint_state['max_val'] - joint_state['min_val']):
            move_flag = True

        if abs(sign) > 0.02 and sign * (joint_value - joint_state['cur_val']) < 0:
            reward *= -1
        joint_state['last_val'] = joint_state['cur_val']
        joint_state['cur_val'] = joint_value
        self.last_move_flag = move_flag
        return reward, move_flag


    def get_joint_type(self):
        joint_info = self.bc.getJointInfo(self.body_id, self.link_id)
        return joint_info[2] == self.bc.JOINT_REVOLUTE


    def get_dist2init(self):
        link_id = self.selected_joint if self.with_goal_state else self.link_id
        joint_state = self.joint_states[link_id]
        dist2init = abs(joint_state['cur_val'] - joint_state['init_val'])

        return dist2init


    def get_init_boundary(self):
        reach_init = False
        reach_boundary = False
        joint_state = self.joint_states[self.link_id]
        joint_info = self.bc.getJointInfo(self.body_id, self.link_id)
        threshold = 0.1 if joint_info[2] == self.bc.JOINT_REVOLUTE else 0.05
        threshold = min(threshold, (joint_state['max_val'] - joint_state['min_val']) / 3.5)

        if joint_state['last_val'] > joint_state['init_val'] and joint_state['cur_val'] < joint_state['init_val'] + threshold:
            reach_init = True
        if joint_state['last_val'] < joint_state['init_val'] and joint_state['cur_val'] > joint_state['init_val'] - threshold:
            reach_init = True
        if joint_state['cur_val'] < joint_state['init_val'] + threshold and joint_state['cur_val'] > joint_state['init_val'] - threshold:
            reach_init = True

        if abs(joint_state['cur_val'] - joint_state['init_val']) > threshold and (joint_state['cur_val'] < joint_state['min_val'] + threshold or joint_state['cur_val'] > joint_state['max_val'] - threshold):
            reach_boundary = True

        return reach_init, reach_boundary


    def get_suction_target_position(self):
        link_state = self.bc.getLinkState(self.body_id, self.link_id)
        link_pos, link_quat = link_state[0], link_state[1]
        suction_position = np.array(self.bc.multiplyTransforms(link_pos, link_quat, self.transform_start_to_link[0], self.transform_start_to_link[1])[0])
        return suction_position


    def _move_to(self, target_position, speed_ratio=1, sleep=False):
        def finetune(target, current):
            while target - current > np.pi:
                target -= np.pi * 2
            while target - current < -np.pi:
                target += np.pi * 2
            return target
        speed_prismatic = 0.002 * speed_ratio
        speed_revolute = 0.05 * speed_ratio
        
        target_position = np.array(target_position) - self._mount_base_position
        self.bc.setJointMotorControlArray(
            self._suction_gripper,
            [0, 1, 2],
            self.bc.POSITION_CONTROL,
            targetPositions=target_position,
            forces=[self._mount_force] * 3,
            positionGains=[speed_prismatic] * 3
        )

        last_position = np.array([-10, -10, -10])
        no_move_cnt = 0

        step_num = 240 * 8
        for i in range(step_num):
            link_state = self.bc.getLinkState(self.body_id, self.link_id)
            link_pos, link_quat = link_state[0], link_state[1]
            target_direction = np.array(self.bc.getMatrixFromQuaternion(link_quat)).reshape([3, 3]) @ self.normal_to_link.T
            theta_z = np.arctan2(target_direction[1], target_direction[0]) + np.pi/2
            theta_x = np.arccos(target_direction[2])
            target_orientation = [
                finetune(theta_z, self.bc.getJointState(self._suction_gripper, 3)[0]),
                finetune(theta_x, self.bc.getJointState(self._suction_gripper, 4)[0])
            ]
            self.bc.setJointMotorControlArray(
                self._suction_gripper,
                [3, 4],
                self.bc.POSITION_CONTROL,
                targetPositions=target_orientation,
                forces=[self._mount_force] * 2,
                positionGains=[speed_revolute] * 2
            )

            self.bc.stepSimulation()

            current_position = np.array([self.bc.getJointState(self._suction_gripper, joint_id)[0] for joint_id in [0, 1, 2]]) + self._mount_base_position
            distance2last = np.sqrt(np.sum((current_position - last_position) ** 2))
            last_position = current_position
            no_move_cnt += int(distance2last < 1e-4)
            if no_move_cnt >= 3:
                break
            
            if sleep:
                time.sleep(3.0 / step_num)


    def get_observation(self):
        self.cam_position = np.array([0, -2, 2])
        self.cam_lookat = np.array([0, 0, 0])
        self.cam_up_direction = np.array([0, 0, 1])
        self.color_image, self.depth_image, self.segmentation_mask, self.cam_pose_matrix, self.cam_view_matrix = \
            self.get_scene_cam_data(self.cam_position, self.cam_lookat, self.cam_up_direction)

        xyz_pts, color_pts, segmentation_pts = utils.get_pointcloud(self.depth_image, self.color_image, self.segmentation_mask, self._scene_cam_intrinsics, self.cam_pose_matrix)
        self.xyz_pts = xyz_pts
        self.body_id_pts = segmentation_pts & ((1 << 24) - 1)
        self.link_id_pts = (segmentation_pts >> 24) - 1
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_pts)
        pcd.colors = o3d.utility.Vector3dVector(color_pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=self.cam_position)
        self.normals = np.array(pcd.normals)

        # crop (remove plane)
        cropped_pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-np.inf, -np.inf, 0.005]), max_bound=np.array([np.inf, np.inf, np.inf])))
        num_points = np.asarray(cropped_pcd.points).shape[0]
        selected_idx = np.random.choice(num_points, min(num_points, 10000), replace=False)
        xyz_pts_sample = np.asarray(cropped_pcd.points)[selected_idx, :]
        color_pts_sample = np.asarray(cropped_pcd.colors)[selected_idx, :]
        normal_pts_sample = np.asarray(cropped_pcd.normals)[selected_idx, :]
        self.pcd = np.concatenate([xyz_pts_sample, color_pts_sample, normal_pts_sample], axis=1).astype(np.float32)

        self.image = np.concatenate([
            self.xyz_pts.reshape(self.color_image.shape),
            self.color_image,
            self.normals.reshape(self.color_image.shape),
            self.depth_image[:, :, np.newaxis]
        ], axis=2).astype(np.float32)

        observation = {
            'image': self.image,
            'pcd': self.pcd,
            'image_init': self.image_init,
            'pcd_init': self.pcd_init,
            'cam_intrinsics': self._scene_cam_intrinsics,
            'cam_pose_matrix': self.cam_pose_matrix,
            'cam_view_matrix': self.cam_view_matrix
        }

        self.image_init = self.image if self.image_init is None else self.image_init
        self.pcd_init = self.pcd if self.pcd_init is None else self.pcd_init

        return observation


    def get_scene_state(self):
        scene_state = {
            'category_name': self.category_name,
            'instance_id': self.instance_id,
            'base_orientation': self.base_orientation,
            'base_position': self.base_position,
            'scaling': self.scaling,
            'with_goal_state': self.with_goal_state,
            'selected_joint': self.selected_joint,
            'joint_states': self.joint_states
        }
        return scene_state

    
    def get_gt_position(self):
        while True:
            random_idx = np.random.choice(len(self.body_id_pts))
            if int(self.body_id_pts[random_idx]) == self.body_id and int(self.link_id_pts[random_idx]) in self.joint_states:
                break
        w, h = np.unravel_index(random_idx, self.depth_image.shape)
        return [w, h]


    def reset(self, scene_state=None, category_type=None, instance_type=None, category_name=None, instance_id=None, with_goal_state=False, **kwargs):
        """Remove all objects; load a new scene; return observation
        Args:
            category_type: train / test
            instance_type: train / test
            category_name: str / None
            instance_id: str / None
            with_goal_state: bool
        Returns:
            observation: dict
        """
        self.bc.resetSimulation()
        self.bc.setGravity(0, 0, 0)
        self._plane_id = self.bc.loadURDF("assets/plane/plane.urdf")
        self._suction_gripper = None

        # Initialization
        self.joint_states = dict()
        self.image_init = None
        self.pcd_init = None

        # Load object
        if scene_state is None:
            self.with_goal_state = with_goal_state
            self.category_name, self.instance_id, urdf_path, orientation, offset, scaling, moveable_link_dict = sim_utils.fetch_mobility_object(category_type, instance_type, category_name, instance_id)
            self.base_orientation = self.bc.multiplyTransforms([0, 0, 0], self.bc.getQuaternionFromEuler([0, 0, np.random.rand() * np.pi/2 - np.pi/4]), [0, 0, 0], orientation)[1]
            self.base_position = scaling * np.array(offset) + np.array([0, -0.5, 0])
            self.scaling = scaling
        else:
            self.with_goal_state = scene_state['with_goal_state']
            self.category_name, self.instance_id = scene_state['category_name'], scene_state['instance_id']
            self.base_position = scene_state['base_position']
            self.base_orientation = scene_state['base_orientation']
            self.scaling = scene_state['scaling']
            urdf_path = os.path.join('mobility_dataset', self.category_name, self.instance_id, 'mobility.urdf')
        urdf_flag = self.bc.URDF_USE_MATERIAL_COLORS_FROM_MTL
        if self.category_name == 'Toy':
            urdf_flag = urdf_flag and self.bc.URDF_USE_SELF_COLLISION_INCLUDE_PARENT and self.bc.URDF_USE_SELF_COLLISION
        
        # Load URDF
        self.body_id = self.bc.loadURDF(
            fileName=urdf_path,
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            globalScaling=self.scaling,
            useFixedBase=True,
            flags=urdf_flag
        )
        self.urdf_path = urdf_path
        
        # Generate scene state
        if scene_state is None:
            # Get min and max
            for joint_id in range(self.bc.getNumJoints(self.body_id)):
                joint_info = self.bc.getJointInfo(self.body_id, joint_id)
                if not joint_info[12].decode('gb2312') in moveable_link_dict:
                    continue
                self.joint_states[joint_id] = {
                    'min_val': joint_info[8],
                    'max_val': joint_info[9]
                }
            self.selected_joint = np.random.choice(list(self.joint_states.keys())) if self.with_goal_state else None
            
            for joint_id in self.joint_states.keys():
                min_val = self.joint_states[joint_id]['min_val']
                max_val = self.joint_states[joint_id]['max_val']
                rand_float = np.random.rand()
                if rand_float < 0.15:
                    cur_val = min_val + (max_val - min_val) * 0.1 # hack: make sure the door is not fully closed
                elif rand_float < 0.3:
                    cur_val = max_val
                else:
                    cur_val = np.random.rand() * (max_val - min_val) + min_val

                self.joint_states[joint_id]['init_val'] = cur_val
                self.joint_states[joint_id]['cur_val'] = cur_val
        else:
            self.selected_joint = scene_state['selected_joint']
            self.joint_states = scene_state['joint_states']
        
        for joint_id in self.joint_states.keys():
            self.bc.resetJointState(self.body_id, joint_id, self.joint_states[joint_id]['init_val'])
        if self.category_name == 'Toy':
            x_distance = np.abs(self.toy_joint_vals[self.instance_id][:, 0] - self.joint_states[2]['init_val'])
            self.bc.resetJointState(self.body_id, 1, self.toy_joint_vals[self.instance_id][np.argmin(x_distance), 1])
        observation = self.get_observation()
        
        if self.with_goal_state:
            self.bc.resetJointState(self.body_id, self.selected_joint, self.joint_states[self.selected_joint]['cur_val'])
            if self.category_name == 'Toy':
                x_distance = np.abs(self.toy_joint_vals[self.instance_id][:, 0] - self.joint_states[2]['cur_val'])
                self.bc.resetJointState(self.body_id, 1, self.toy_joint_vals[self.instance_id][np.argmin(x_distance), 1])
            observation = self.get_observation()

        return observation


    def step(self, action, **kwargs):
        """Execute action and return reward and next observation.

        Args:
            action: [action_type=0, w, h] / [action_type=1, x, y, z]
                Case 0: Choose action position.
                Case 1: Choose direction.
        
        Returns:
            observation: 
                image:      [W, H, 10], np.float32 ==> [xyz(3), color(3), normal(3), depth(1)]
                pcd:        [N, 9], np.float32 ==> [xyz(3), color(3), normal(3)]
                image_init
                pcd_init
                cam_intrinsics
                cam_pose_matrix
                cam_view_matrix
            reward: bool + float 
                bool: moveable
                foat: signed distance
            done: 
                position: terminate (bool)
                direction: reach_boundary (bool) + reach_init (bool)
            info: dict
        """
        action_type = action[0]
        if action_type == 0:
            pixel_index = np.ravel_multi_index(action[1:], self.depth_image.shape)

            body_id = int(self.body_id_pts[pixel_index])
            link_id = int(self.link_id_pts[pixel_index])

            # wrong position, terminate immediately
            if not (body_id == self.body_id and link_id in self.joint_states):
                return self.get_observation(), (0, False), True, dict()
            if self.with_goal_state and link_id != self.selected_joint:
                return self.get_observation(), (0, False), True, dict()

            position_start = self.xyz_pts[pixel_index]
            surface_normal = self.normals[pixel_index]

            self._suction_gripper = self.bc.loadURDF(
                self.ee_urdf,
                useFixedBase=True,
                basePosition=self._mount_base_position,
                globalScaling=1,
                flags=self.bc.URDF_USE_MATERIAL_COLORS_FROM_MTL
            )
            link_state = self.bc.getLinkState(body_id, link_id)
            vec_inv, quat_inv = self.bc.invertTransform(link_state[0], link_state[1])
            self.transform_start_to_link = self.bc.multiplyTransforms(vec_inv, quat_inv, position_start, [0, 0, 0, 1])
            self.normal_to_link = np.array(self.bc.getMatrixFromQuaternion(quat_inv)).reshape([3, 3]) @ np.array(surface_normal).T
            self.link_id = link_id

            self._move_to(self._mount_base_position, speed_ratio=10)
            self._move_to(position_start, speed_ratio=10)

            constraint_id = self.bc.createConstraint(
                parentBodyUniqueId=body_id,
                parentLinkIndex=link_id,
                childBodyUniqueId=self._suction_gripper,
                childLinkIndex=2,
                jointType=self.bc.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=self.transform_start_to_link[0],
                parentFrameOrientation=self.transform_start_to_link[1],
                childFramePosition=[0, 0, 0],
            )
            self.bc.changeConstraint(constraint_id, maxForce=self._constraint_force)

            info = {
                'dist2init': self.get_dist2init(),
                'position_start': position_start,
                'surface_normal': surface_normal
            }

            return self.get_observation(), (1, False), False, info

        elif action_type == 1:
            direction_vec = np.array(action[1:4])
            position_start = self.get_suction_target_position()
            position_end = position_start + direction_vec * self._action_distance

            # visualize actions if gui is enabled
            if self.gui:
                self.bc.addUserDebugLine(lineFromXYZ=position_start, lineToXYZ=position_start + direction_vec * self._action_distance * 20, lineColorRGB=(0, 0, 1))
                self.bc.addUserDebugLine(lineFromXYZ=position_start, lineToXYZ=position_end, lineColorRGB=(0, 1, 1))
            
            self._move_to(position_end, sleep=self.gui, speed_ratio=1)

            # this step is to make the ee aligned again
            self._move_to(self.get_suction_target_position(), sleep=self.gui, speed_ratio=10)
            
            if self.gui:
                self.bc.removeAllUserDebugItems()
            

            reward, move_flag = self.get_reward()
            reach_init, reach_boundary = self.get_init_boundary()
            observation = self.get_observation()
            info = {
                'position_start': position_start,
                'position_end': position_end,
                'dist2init': self.get_dist2init()
            }

            return observation, (reward, move_flag), (reach_init, reach_boundary), info


if __name__=='__main__':
    import spherical_sampling
    direction_candidates = spherical_sampling.fibonacci(64, co_ords='cart')

    seed = 123
    sim = PybulletSim(False, 0.18)

    observation = sim.reset(category_type='train', instance_type='train')

    # get GT position
    w, h = sim.get_gt_position()
    observation, reward, terminate, info = sim.step([0, w, h])

    for i in range(5):
        # random direction
        direction = direction_candidates[np.random.choice(len(direction_candidates))]
        observation, reward, (reach_init, reach_boundary), info = sim.step([1, direction[0], direction[1], direction[2]])