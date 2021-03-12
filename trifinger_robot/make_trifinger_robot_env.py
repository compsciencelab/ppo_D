
import os
import gym
import numpy as np
from baselines import bench
import sys
sys.path.append('.')
import rrc_simulation
from rrc_simulation.tasks import move_cube
from rrc_simulation import visual_objects
from rrc_simulation import TriFingerPlatform
from rrc_simulation.gym_wrapper.envs import cube_env
from trifinger_robot.ppod_wrappers import TrifingerReplayRecord


from trifinger_robot.wrappers import FlatObservationWrapper, ActionWrapper, ActionClipper

def make_trifinger_robot_env(log_dir=None,base_seed= 0, frame_skip=0, allow_early_resets=False, visualization=False, initializer=None,difficulty=3,delta=0.1, action=False, rho=0.1, phi=0.3, size_buffer=50, size_buffer_V=50, demo_dir= '', threshold_reward=None):

    def make_env(rank):

        if log_dir and not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        def _thunk():

            env = ExamplePushingTrainingEnv(
                initializer=initializer,
                action_type=cube_env.ActionType.POSITION,
                frameskip=frame_skip,
                visualization=visualization,
                difficulty=difficulty,
                delta=delta
            )

            env.seed(base_seed + rank)
            env.action_space.seed(seed=base_seed + rank)
            env = ActionClipper(env)
            env = TrifingerReplayRecord(env,rho, phi, demo_dir, size_buffer, size_buffer_V, threshold_reward)
            

            if action: 
                env = ActionWrapper(env)

            env = FlatObservationWrapper(env)

            if log_dir is not None:
                env = bench.Monitor(
                    env, os.path.join(
                        log_dir, "{}_{}".format(base_seed, str(rank))),
                    allow_early_resets=allow_early_resets,
                    info_keywords=('r0','r1','goal', 'goal_coor', 'goal_or', 'min_r0', 'ereward','reward_woD'))

            return env

        return _thunk

    return make_env

def reward_sigmoid(x, threshold = 0.1, sign=-1):
    # sign determines the direction of the step, negative is lower than threshold goes to one
    sigma = 1 # lower somooth, higher sharp, 1 normal
    z = 1/(1 + np.exp(sign*(sigma*x -threshold)))
    return z

class ExamplePushingTrainingEnv(gym.Env):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        initializer,
        action_type,
        frameskip,
        visualization,
        difficulty,
        delta
    ):
        """Initialize.

        Args:
            initializer: Initializer class for providing initial cube pose and
                goal pose. If no initializer is provided, we will initialize in a way
                which is be helpful for learning.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================

        self.initializer = initializer
        self.action_type = action_type
        self.visualization = visualization
        self.difficulty=difficulty
        self.delta = delta

        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        spaces = TriFingerPlatform.spaces

        if self.action_type == cube_env.ActionType.TORQUE:
            self.action_space = spaces.robot_torque.gym
        elif self.action_type == cube_env.ActionType.POSITION:
            self.action_space = spaces.robot_position.gym
        elif self.action_type == cube_env.ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
        else:
            raise ValueError("Invalid action_type")

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": spaces.robot_position.gym,
                "robot_velocity": spaces.robot_velocity.gym,
                "robot_tip_positions": gym.spaces.Box(
                    low=np.array([spaces.object_position.low] * 3),
                    high=np.array([spaces.object_position.high] * 3),
                ),
                "object_position": spaces.object_position.gym,
                "object_orientation": spaces.object_orientation.gym,
                "goal_object_position": spaces.object_position.gym,
            }
        )
        
        if self.initializer is None:
            # if no initializer is given (which will be the case during training),
            # we can initialize in any way desired. here, we initialize the cube always
            # in the center of the arena, instead of randomly, as this appears to help
            # training
            self.initial_robot_position = TriFingerPlatform.spaces.robot_position.default
            #default_object_position = TriFingerPlatform.spaces.object_position.default
            #default_object_orientation = TriFingerPlatform.spaces.object_orientation.default
            #initial_object_pose = move_cube.Pose(position=default_object_position,orientation=default_object_orientation,)
            self.initial_object_pose = move_cube.sample_goal(-1)
            self.goal_object_pose = move_cube.sample_goal(difficulty=self.difficulty) 
        else:
            # if an initializer is given, i.e. during evaluation, we need to initialize
            # according to it, to make sure we remain coherent with the standard CubeEnv.
            # otherwise the trajectories produced during evaluation will be invalid.
            self.initial_robot_position = TriFingerPlatform.spaces.robot_position.default
            self.initial_object_pose=self.initializer.get_initial_state()
            self.goal_object_pose = self.initializer.get_goal()


       
        self.initial_object_pose.position = np.array([ 0.0, 0.0,  0.0325    ])
        self.initial_object_pose.orientation =  np.array([0, 0, 0, 1])
        #self.initial_robot_position = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  
        self.goal_object_pose.position =  np.array([0,  0,  0.1])
        self.goal_object_pose.orientation = np.array([0, 0, 0, 1])

            

    def step(self, action):
        
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        old_reward = 0.0
        r0 = 0.0
        r1 = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            previous_observation = self._create_observation(t)
            observation = self._create_observation(t + 1)

            # OLD REWARD
            old_reward += self._compute_reward(	
                previous_observation=previous_observation,	
                observation=observation,	
            )

            #NEW REWARD
            reward_func = move_cube.evaluate_state(
                move_cube.Pose(self.goal['position'],self.goal['orientation']),
                move_cube.Pose(observation['object_position'],observation['object_orientation']),
                self.difficulty)

            r0 += reward_func 
            max_dist = np.sqrt( (2.0*move_cube._ARENA_RADIUS)**2 + (move_cube._max_height)**2)
            mean_dist_3tips = np.mean(np.linalg.norm(observation["robot_tip_positions"] - observation["object_position"],
                                                        axis=1,keepdims=True))
            r1 += mean_dist_3tips/max_dist  #normalize to max size in arena
            self.min_r0 = min(self.min_r0, r0)   
                 
        #reward = np.copy(r0)
        r0 = r0/num_steps
        r1 = r1/num_steps
        
        
        if r0<self.delta:
            goal = 1
            print(r0, self.delta, self.min_r0)
        else:
            goal = 0
        #print('RO', r0, self.delta,'GOAL', goal)
        #if goal == 1:
        #    print('----------------------------------------------------------------')
        #reward =  goal + 0.001*(1-r1)
        #reward = old_reward 
        #reward = -np.copy(r0)
        reward= 0.0
        reward += reward_sigmoid(r0)*2
        reward += reward_sigmoid(r1)
        reward += reward_sigmoid(observation["object_position"][2], sign = 1)
        
        
        self.info['r0']=r0  #test reward within frameskip
        self.info['r1']=r1  #other
        self.info['goal']=goal
        self.info['goal_or'] = self.goal_object_pose.orientation
        self.info['goal_coor'] = self.goal_object_pose.position
        self.info['min_r0'] = self.min_r0
        is_done = self.step_count == move_cube.episode_length
        return observation, reward, is_done, self.info

       
    def reset(self):
        # reset simulation
        self.min_r0 = 9999999.0
        del self.platform

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=self.initial_robot_position,
            initial_object_pose=self.initial_object_pose,
        )
        
        self.goal = {
            "position": self.goal_object_pose.position,
            "orientation": self.goal_object_pose.orientation,
        }
        # visualize the goal
        if self.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=0.065,
                position=self.goal_object_pose.position,
                orientation=self.goal_object_pose.orientation,
            )

        self.info = dict()

        self.step_count = 0

        return self._create_observation(0)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        object_observation = self.platform.get_object_pose(t)
        robot_tip_positions = self.platform.forward_kinematics(
            robot_observation.position
        )
        robot_tip_positions = np.array(robot_tip_positions)

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_tip_positions": robot_tip_positions,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "goal_object_position": self.goal["position"],
        }
        return observation

    @staticmethod
    def _compute_reward(previous_observation, observation):

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(
            observation["robot_tip_positions"] - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            previous_observation["robot_tip_positions"]
            - previous_observation["object_position"]
        )

        reward_term_1 = (
            previous_distance_from_block - current_distance_from_block
        )

        # calculate second reward term
        current_dist_to_goal = np.linalg.norm(
            observation["goal_object_position"]
            - observation["object_position"]
        )
        previous_dist_to_goal = np.linalg.norm(
            previous_observation["goal_object_position"]
            - previous_observation["object_position"]
        )
        reward_term_2 = previous_dist_to_goal - current_dist_to_goal

        reward = 250 * reward_term_1 + 750 * reward_term_2

        return reward

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == cube_env.ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == cube_env.ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == cube_env.ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action