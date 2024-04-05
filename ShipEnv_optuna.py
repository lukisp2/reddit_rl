import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import pygame
import sys
import torch
import os
import optuna

class Ship2DEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode='human'):
        super(Ship2DEnv, self).__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            'desired_goal': spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
        })


        self.reward = 0
        self.done = False

        self.max_steps = 100  # Maksymalna liczba kroków w epizodzie
        self.current_step = 0  # Licznik kroków

        self.goal = self._sample_goal()
        self.state = np.zeros(6, dtype=np.float32)
        self.x = 0
        self.y = 0
        self.scale = 1
        self.screen_width = 640 * self.scale
        self.screen_height = 640 * self.scale

        print(self.screen_width, self.screen_height)

#        self.background_image = pygame.image.load('G:/Mój dysk/AKADEMIA/nauka/artykuły moje/RL/Map_150.bmp')
#        self.background_image = pygame.transform.scale(self.background_image, (self.screen_width, self.screen_height))

        self.min_dis_to_target = 50

        self.info = {
            'is_success': self._is_success(self.state, self.goal),
        }

        if self.render_mode == 'human':
            pygame.init()
            self.screen_size = (self.screen_width, self.screen_height)
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Simple Ship Environment")
            self.clock = pygame.time.Clock()

    def normalize_state(self, value,max_,min_) -> float:
        return (value - min_)/(max_ - min_)

    def denormalize_state(self,value,max,min) -> float:
        return value * (max - min) + min

    def _sample_goal(self):
        return np.random.uniform(0, 1, size=6)

    def reset(self, **kwargs):
        self.current_step = 0  # Resetowanie licznika kroków przy każdym resecie
        self.x = np.random.uniform(0, self.screen_width, 1)
        self.y = np.random.uniform(0, self.screen_height, 1)
        self.x = self.screen_width/2
        self.y = self.screen_height/2
        x_norm = self.normalize_state(self.x, self.screen_width, 0)
        y_norm = self.normalize_state(self.y, self.screen_height, 0)

        self.state = np.array([x_norm, y_norm, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.goal = self._sample_goal()
        #self.goal = [0.9, 0.9, 0.0, 0.0, 1.0, 0.0] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print("New goal: ", self.goal)

        self.reward = 0
        self.current_step = 0
        self.done = False
        self.info = {
            'is_success': self._is_success(self.state, self.goal),
        }
        return self._get_obs(), self.info  # Zwracanie początkowej obserwacji

    def _get_obs(self):
        return {
            'observation': self.state.copy(),
            'achieved_goal': self.state.copy(),
            'desired_goal': self.goal.copy(),
        }

    def step(self, action):
        self.current_step += 1  # Inkrementacja licznika kroków
        action = np.array([action[0]*10, action[1] * 10], dtype=np.float32)

        delta_hdg, v = action

        hdg = self.denormalize_state(self.state[2],360,0)
        hdg = hdg + delta_hdg
        #v = self.denormalize_state(v,10,0)

        delta_x = v * np.cos(np.deg2rad(hdg))
        delta_y = v * np.sin(np.deg2rad(hdg))
        self.x = self.x + delta_x
        self.y = self.y + delta_y

        hdg_norm = self.normalize_state(hdg,360,0)
        v = self.normalize_state(v,10,0)

        self.state[0] = self.normalize_state(self.x,self.screen_width,0)
        self.state[1] = self.normalize_state(self.y,self.screen_height,0)
        self.state[2] = hdg_norm
        self.state[3] = v
        self.state[4] = self.normalize_state(np.cos(np.deg2rad(hdg)), 1, 0)
        self.state[5] = self.normalize_state(np.sin(np.deg2rad(hdg)), 1, 0)

        info = {
            'is_success': self._is_success(self.state, self.goal),
        }



        self.reward = self.compute_reward(self.state, self.goal, info)
        self.done = self.current_step >= self.max_steps  # Zakończenie epizodu, jeśli osiągnięto maksymalną liczbę kroków

        if info["is_success"] == True:
            self.done = True

        #print(reward)

        if (self.x >= self.screen_width*self.scale) or (self.y >= self.screen_height*self.scale) or (self.x <= 0)  or (self.y <= 0):
            self.done = True
            info = {
                'is_success':False,
            }
            self.reward = self.reward -1.0

        return self._get_obs(), self.reward, self.done, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal[:1] - desired_goal, axis=-1)
        return -distance

    def _is_success(self, achieved_goal, desired_goal):
        x_now = self.denormalize_state(achieved_goal[0],self.screen_width,0)
        y_now = self.denormalize_state(achieved_goal[1],self.screen_height,0)

        x_goal = self.denormalize_state(desired_goal[0],self.screen_width,0)
        y_goal = self.denormalize_state(desired_goal[1],self.screen_height,0)

        pos = np.array([x_now, y_now])
        pos_goal = np.array([x_goal, y_goal])

        dist = np.linalg.norm(pos - pos_goal, axis=-1)

        return dist < self.min_dis_to_target

    def rotate_point(self, point, angle, center_point):
        """Obraca punkt wokół innego punktu o dany kąt."""
        temp_point = point[0] - center_point[0], point[1] - center_point[1]
        temp_point = (temp_point[0] * np.cos(angle) - temp_point[1] * np.sin(angle),
                      temp_point[0] * np.sin(angle) + temp_point[1] * np.cos(angle))
        temp_point = temp_point[0] + center_point[0], temp_point[1] + center_point[1]
        return temp_point

    def render(self):
        if self.render_mode is None:
            print("Render mode not specified")
            return

        # Initialize Pygame display if it hasn't been already
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (int(self.screen_width), int(self.screen_height))
            )

        # Initialize a clock if it hasn't been already
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Process Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Allows closing the window manually
                self.close()
                return

        # Ensure there's a state to render
        if self.state is None:
            return None

        # Blit the background
        #self.screen.blit(self.background_image, (0, 0))
        target_color = (255, 0, 0)  # Czerwony kolor dla obszaru docelowego
        x_goal = self.denormalize_state(self.goal[0],self.screen_width,0)
        y_goal = self.denormalize_state(self.goal[1],self.screen_height,0)

        pygame.draw.circle(self.screen, target_color, (x_goal, y_goal), self.min_dis_to_target / 2, 2)

        # Ship positioning and rotation
        ship_x = int(self.denormalize_state(self.state[0], self.screen_width * self.scale, 0))
        ship_y = int(self.denormalize_state(self.state[1], self.screen_height * self.scale, 0))
        ship_size = 5
        hdg = self.denormalize_state(self.state[2], 360, 0)
        hdg = np.deg2rad(hdg + 90)  # Convert heading to radians

        # Define base points for the ship
        base_points = [
            (ship_x, ship_y - ship_size),  # Top
            (ship_x - ship_size, ship_y + ship_size),  # Bottom left
            (ship_x, ship_y + ship_size / 2),  # Center
            (ship_x + ship_size, ship_y + ship_size),  # Bottom right
            (ship_x, ship_y - ship_size)  # Back to top to close the polygon
        ]

        # Rotate points around the ship's center
        rotated_points = [self.rotate_point(p, hdg, (ship_x, ship_y)) for p in base_points]

        # Draw the ship on the screen
        pygame.draw.polygon(self.screen, (0, 0, 255), rotated_points)

        # Update the display
        pygame.display.flip()

        # Control the frame rate to 60fps
        self.clock.tick(60)

        # Additional handling for different render modes if necessary
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()




class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0,  redner_interval: int = 0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.render_interval = redner_interval
        self.log_dir = log_dir
        self.writer = None
    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def _on_step(self) -> bool:

        if self.n_calls % self.render_interval == 0:
            self.training_env.render()

        # Przykład logowania actor_loss
        actor_loss = self.locals.get('actor_loss')
        reward = self.locals.get('rewards')
        if actor_loss is not None:
            self.writer.add_scalar('loss/actor_loss', actor_loss, self.num_timesteps)
            self.writer.add_scalar('loss/reward', reward,self.num_timesteps)
        # Dodaj tutaj logowanie innych metryk w podobny sposób

        return True

    def _on_training_end(self) -> None:
        self.writer.close()


log_dir = "C:/tensor_board_logs/30.03.2024"
#device = torch.device('cpu')
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def optimize_model(trial):
    # Hiperparametry do strojenia
    buffer_size = trial.suggest_categorical('buffer_size', [100000])
    batch_size = trial.suggest_categorical('batch_size', [64,128])
    gamma = trial.suggest_loguniform('gamma', 0.95, 0.99)
    learning_rate = trial.suggest_loguniform('learning_rate', 6.7e-4, 8.5e-4)
    net_arch = trial.suggest_categorical('net_arch', [[1024, 1024, 1024],[2048,2048,2048],[1024,1024,1024,1024]])

    model = SAC(
        "MultiInputPolicy",
        env,
        buffer_size=buffer_size,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            copy_info_dict=True
        ),
        verbose=1,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=log_dir,
        learning_starts=8000
    )

    model.learn(total_timesteps=1500000)  # Zmień na odpowiednią liczbę kroków
    rewards = evaluate_model(model)  # Ta funkcja musi być zdefiniowana, aby ocenić model

    return rewards


def evaluate_model(model):
    # Ta funkcja powinna zwrócić ocenę modelu, np. średnią nagrodę z N epizodów
    # Implementacja zależy od twojego środowiska i tego, jak chcesz ocenić model
    total_rewards = 0
    for _ in range(10):  # Przykładowa liczba epizodów do oceny
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            done = dones
            total_rewards += rewards
    return total_rewards / 10  # Zwraca średnią nagrodę


# Przygotowanie środowiska
env = Ship2DEnv(render_mode='human')
env = DummyVecEnv([lambda: env])

# Uruchomienie procesu strojenia
study = optuna.create_study(direction='maximize')
study.optimize(optimize_model, n_trials=20)  # Możesz zwiększyć liczbę prób

print('Best trial:')
trial = study.best_trial

print(f'Value: {trial.value}')
print('Params: ')
for key, value in trial.params.items():
    print(f'{key}: {value}')

