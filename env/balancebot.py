import os
import collections
import numpy as np
class BalanceBot(object):
    
    def __init__(self, 
                pybullet_client, 
                time_step, 
                action_repeat, 
                control_latency):

        self._p = pybullet_client

        self._motor_strength = {"torso_l_wheel": 0.0, "torso_r_wheel": 0.0}
        self._motor_direction = {"torso_l_wheel": -1, "torso_r_wheel": 1}
       
        self._action_repeat = action_repeat
        self._max_pwm = (5000. / 3200.) * (2.0 * 3.1415926)
        self._max_force = 0.4

        self._time_step = time_step
        self._control_latency = control_latency
        self._observation_history = collections.deque(maxlen=5)
        self._delayed_observation = []


    def _buildJointNameToIdDict(self):

        num_joints = self._p.getNumJoints(self.balancebot_id)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._p.getJointInfo(self.balancebot_id, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        

    def _buildUrdfIds(self):

        self._base_link_id = -1
        self._wheel_link_id = [0, 1]
        self._wheel_joint_id = [0, 1]

    def reset(self):
        
        random_init_angle = [np.random.uniform(-0.05, 0.05),
                        np.random.uniform(-0.01, 0.01),
                        np.random.uniform(-0.05, 0.05)]
                        
        random_init_orient = self._p.getQuaternionFromEuler(random_init_angle)

        urdf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "balancebot_simple.urdf")
        self.balancebot_id = self._p.loadURDF(urdf_file, 
                                                basePosition=[0, 0, 0.01],
                                                baseOrientation=random_init_orient)

        self._buildJointNameToIdDict()
        self._buildUrdfIds()

        for wheel in ['torso_l_wheel', 'torso_r_wheel']:
            wheelid = self._joint_name_to_id[wheel]
            self._p.setJointMotorControl2(self.balancebot_id,wheelid,self._p.VELOCITY_CONTROL,targetVelocity=0,force=0)

        self._observation_history.clear()
        self.receiveObservation()
        
        return None


    def step(self, action):

        for _ in range(self._action_repeat):
            self.applyAction(action)
            self._p.stepSimulation()
            self.receiveObservation()

    def _getTrueObservation(self):

        basePos, baseOrn = self._p.getBasePositionAndOrientation(self.balancebot_id)
        baseRPY = self._p.getEulerFromQuaternion(baseOrn)
        baseLinVel, baseAngVel = self._p.getBaseVelocity(self.balancebot_id)
        observation = [basePos, baseLinVel, baseRPY, baseAngVel]
        
        return observation

    def receiveObservation(self):

        self._observation_history.appendleft(self._getTrueObservation())
        self._delayed_observation = self._getDelayedObservation(self._control_latency)


    def _getDelayedObservation(self, latency):
        
        if latency <= 0 or len(self._observation_history) == 1:
            observation = self._observation_history[0]
        else:
            n_steps_ago = int(latency / self._time_step)
            if n_steps_ago + 1 >= len(self._observation_history):
                return self._observation_history[-1]
            remaining_latency = latency - n_steps_ago * self._time_step
            blend_alpha = remaining_latency / self._time_step
            observation = (
                (1.0 - blend_alpha) * np.array(self._observation_history[n_steps_ago])
                + blend_alpha * np.array(self._observation_history[n_steps_ago + 1]))
        return observation


    def applyAction(self, motor_cmd):

        assert type(motor_cmd) == np.ndarray

        motor_cmd = np.clip(motor_cmd[0], -1.0, 1.0)
        motor_name_list = ['torso_l_wheel', 'torso_r_wheel']
        
        for motor_name in motor_name_list:

            self._motor_strength[motor_name] = motor_cmd

            self._actual_write_cmd = np.clip(self._motor_strength[motor_name], -1.0, 1.0)
            self._actual_motor_pwm = self._actual_write_cmd * self._max_pwm * self._motor_direction[motor_name] 
            self._setMotorVelocityById(self._joint_name_to_id[motor_name], self._actual_motor_pwm)
        

    def _setMotorVelocityById(self, motor_id, motor_vel):
        self._p.setJointMotorControl2(
            bodyUniqueId=self.balancebot_id, 
            jointIndex=motor_id,
            controlMode=self._p.VELOCITY_CONTROL,
            targetVelocity=motor_vel,
            force=self._max_force)

    def getBasePosition(self):
        delayed_position = np.array(self._delayed_observation[0])
        return delayed_position

    def getBaseLinVelocity(self):
        delayed_lin_vel = np.array(self._delayed_observation[1])
        return delayed_lin_vel

    def getBaseRollPitchYaw(self):
        delayed_roll_pitch_yaw = np.array(self._delayed_observation[2])
        return delayed_roll_pitch_yaw

    def getMotorStrength(self):

        l_motor = self._motor_strength["torso_l_wheel"]
        r_motor = self._motor_strength["torso_r_wheel"]

        return np.array([l_motor, r_motor])

