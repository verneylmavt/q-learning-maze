#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
import random


# In[10]:


class MazeEnv:
    def __init__(self, n_rows, n_cols, start, goal, obstacle_prob=0.2, seed=None):
        """
        n_rows, n_cols: Dimensions of the maze.
        start, goal: Tuple coordinates (row, col).
        obstacle_prob: Probability for a cell to be a wall.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal
        self.obstacle_prob = obstacle_prob
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.generate_maze()
        self.current_state = start
        
        self.actions = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }
        
    def generate_maze(self):
        """Generates a maze grid with random obstacles. Walls are represented by 1; free cells by 0."""
        self.maze = np.zeros((self.n_rows, self.n_cols), dtype=int)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if (i, j) == self.start or (i, j) == self.goal:
                    continue
                if np.random.rand() < self.obstacle_prob:
                    self.maze[i, j] = 1
        
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        self.maze[self.start] = 0
        self.maze[self.goal] = 0
        
    def reset(self):
        """Resets the environment and returns the start state."""
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        """
        Executes the given action.
        Returns: next_state, reward, done (boolean).
        """
        delta = self.actions[action]
        next_state = (self.current_state[0] + delta[0],
                      self.current_state[1] + delta[1])
        
        if (0 <= next_state[0] < self.n_rows) and (0 <= next_state[1] < self.n_cols):
            if self.maze[next_state] != 1:
                self.current_state = next_state
            else:
                next_state = self.current_state
        else:
            next_state = self.current_state
            
        if next_state == self.goal:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
            
        return next_state, reward, done
    
    def render(self, path=None):
        """
        Visualizes the maze using Plotly.
        """
        fig = go.Figure()
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                color = 'white'
                if self.maze[i, j] == 1:
                    color = 'black'
                if (i, j) == self.start:
                    color = 'green'
                if (i, j) == self.goal:
                    color = 'red'
                if path is not None and (i, j) in path:
                    color = 'blue'
                    
                fig.add_shape(
                    type="rect",
                    x0=j, y0=self.n_rows - i - 1,
                    x1=j+1, y1=self.n_rows - i,
                    line=dict(color="gray"),
                    fillcolor=color,
                )
        fig.update_xaxes(showticklabels=False, range=[0, self.n_cols])
        fig.update_yaxes(showticklabels=False, range=[0, self.n_rows])
        fig.update_layout(width=500, height=500, title="Maze")
        fig.show()


# In[11]:


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        env: MazeEnv instance.
        learning_rate: α in Q-learning update.
        discount: γ in Q-learning update.
        epsilon: Initial exploration rate.
        epsilon_decay: Multiplicative decay per episode.
        min_epsilon: Minimum exploration rate.
        """
        self.env = env
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((env.n_rows, env.n_cols, 4))
        
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        else:
            row, col = state
            return np.argmax(self.q_table[row, col])

    def train(self, episodes=500, max_steps=1000, verbose=False):
        """
        Trains the agent for a number of episodes.
        Returns a list of total rewards per episode.
        """
        rewards_per_episode = []
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                row, col = state
                n_row, n_col = next_state
                best_next = np.max(self.q_table[n_row, n_col])
                self.q_table[row, col, action] += self.lr * (
                    reward + self.discount * best_next - self.q_table[row, col, action]
                )
                state = next_state
                total_reward += reward
                if done:
                    break
            rewards_per_episode.append(total_reward)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if verbose and (ep+1) % 100 == 0:
                print(f"Episode {ep+1}: Total Reward = {total_reward}")
        return rewards_per_episode
    
    def get_optimal_path(self, max_steps=1000):
        """
        Returns the optimal (greedy) path starting from the start state,
        following the learned Q-values.
        """
        path = []
        state = self.env.reset()
        path.append(state)
        for _ in range(max_steps):
            row, col = state
            action = np.argmax(self.q_table[row, col])
            next_state, reward, done = self.env.step(action)
            if next_state == state:
                break
            path.append(next_state)
            state = next_state
            if done:
                break
        return path


# In[12]:


maze_rows_widget = widgets.IntSlider(value=10, min=5, max=30, description='Rows')
maze_cols_widget = widgets.IntSlider(value=10, min=5, max=30, description='Cols')
start_row_widget = widgets.IntText(value=1, description='Start Row')
start_col_widget = widgets.IntText(value=1, description='Start Col')
goal_row_widget = widgets.IntText(value=8, description='Goal Row')
goal_col_widget = widgets.IntText(value=8, description='Goal Col')
obstacle_prob_widget = widgets.FloatSlider(value=0.2, min=0.0, max=0.5, step=0.05, description='Obs Prob')

episodes_widget = widgets.IntSlider(value=500, min=100, max=2000, step=100, description='Episodes')
learning_rate_widget = widgets.FloatSlider(value=0.1, min=0.01, max=1.0, step=0.01, description='Learning Rate')
discount_widget = widgets.FloatSlider(value=0.95, min=0.5, max=1.0, step=0.01, description='Discount')
epsilon_widget = widgets.FloatSlider(value=1.0, min=0.1, max=1.0, step=0.05, description='Epsilon')
epsilon_decay_widget = widgets.FloatSlider(value=0.995, min=0.90, max=1.0, step=0.001, description='Epsilon Decay')
max_steps_widget = widgets.IntSlider(value=1000, min=100, max=5000, step=100, description='Max Steps')


# In[13]:


train_button = widgets.Button(description="Train Q-Learning Agent", button_style='success')


# In[14]:


def on_train_button_clicked(b):
    clear_output(wait=True)
    
    display(ui)
    
    n_rows = maze_rows_widget.value
    n_cols = maze_cols_widget.value
    start = (start_row_widget.value, start_col_widget.value)
    goal = (goal_row_widget.value, goal_col_widget.value)
    obstacle_prob = obstacle_prob_widget.value
    
    episodes = episodes_widget.value
    learning_rate = learning_rate_widget.value
    discount = discount_widget.value
    epsilon = epsilon_widget.value
    epsilon_decay = epsilon_decay_widget.value
    max_steps = max_steps_widget.value
    
    env = MazeEnv(n_rows, n_cols, start, goal, obstacle_prob)
    
    print("### Initial Maze ###")
    env.render()
    
    agent = QLearningAgent(env, learning_rate, discount, epsilon, epsilon_decay)
    
    print("### Training Agent... ###")
    rewards = agent.train(episodes, max_steps, verbose=True)
    
    path = agent.get_optimal_path(max_steps)
    print("### Learned Path ###")
    print(path)
    
    env.render(path)


# In[15]:


train_button.on_click(on_train_button_clicked)

maze_params = widgets.VBox([
    widgets.Label("Maze Parameters:"),
    maze_rows_widget, 
    maze_cols_widget, 
    start_row_widget, 
    start_col_widget,
    goal_row_widget, 
    goal_col_widget,
    obstacle_prob_widget
])
learning_params = widgets.VBox([
    widgets.Label("Q-Learning Parameters:"),
    episodes_widget, 
    learning_rate_widget, 
    discount_widget, 
    epsilon_widget, 
    epsilon_decay_widget, 
    max_steps_widget
])
ui = widgets.HBox([maze_params, learning_params, train_button])

display(ui)

