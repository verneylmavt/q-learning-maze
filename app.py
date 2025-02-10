import streamlit as st
from streamlit_extras.mention import mention
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Class Definition
# -----------------------------

class MazeEnv:
    def __init__(self, n_rows, n_cols, start, goal, obstacle_prob=0.2, seed=None):
        """
        n_rows, n_cols: dimensions of the maze.
        start, goal: Tuple coordinates (row, col).
        obstacle_prob: probability for a cell (not start/goal) to be a wall.
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
        """Generates a maze grid with random obstacles.
           Walls are represented by 1; free cells by 0.
        """
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
        """Resets the environment (starting position) and returns the start state."""
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        """
        Executes the given action.
        Returns: next_state, reward, done.
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
        Returns a Plotly figure visualizing the maze.
        If a list of coordinates 'path' is provided, those cells are colored.
        """
        fig = go.Figure()
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                color = '#eff2f6'
                if self.maze[i, j] == 1:
                    color = '#262730'
                if (i, j) == self.start:
                    color = '#00ffd5'
                if (i, j) == self.goal:
                    color = '#ffabab'
                if path is not None and (i, j) in path:
                    color = '#0068c9'
                    
                fig.add_shape(
                    type="rect",
                    x0=j, y0=self.n_rows - i - 1,
                    x1=j+1, y1=self.n_rows - i,
                    line=dict(color="gray"),
                    fillcolor=color,
                )
        fig.update_xaxes(showticklabels=False, range=[0, self.n_cols])
        fig.update_yaxes(showticklabels=False, range=[0, self.n_rows])
        fig.update_layout(width=800, height=800)
        return fig


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        env: MazeEnv instance.
        learning_rate: α in Q-learning.
        discount: γ in Q-learning.
        epsilon: initial exploration rate.
        epsilon_decay: multiplicative decay per episode.
        min_epsilon: minimum exploration rate.
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
    
    def train(self, episodes=500, max_steps=1000):
        """
        Trains the agent.
        Returns a list of total rewards per episode.
        """
        rewards_per_episode = []
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for _ in range(max_steps):
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
        return rewards_per_episode
    
    def get_optimal_path(self, max_steps=1000):
        """
        Returns the optimal (greedy) path starting from the start state.
        """
        path = []
        state = self.env.reset()
        path.append(state)
        for _ in range(max_steps):
            row, col = state
            action = np.argmax(self.q_table[row, col])
            next_state, _, done = self.env.step(action)
            if next_state == state:
                break
            path.append(next_state)
            state = next_state
            if done:
                break
        return path

# -----------------------------
# Page UI
# -----------------------------

def main():
    st.set_page_config(page_title="Maze Solver w/ Q-Learning"
                    # layout="wide"
                    )
    
    st.title("Maze Solver w/ Q-Learning")
    st.divider()
    
    with st.container(border=True):
        cola, colb = st.columns(2)
        with cola:
            rows = st.number_input("Number of Rows", min_value=5, max_value=50, value=10, step=1)
        with colb:
            cols = st.number_input("Number of Columns", min_value=5, max_value=50, value=10, step=1)
        obstacle_prob = st.slider("Probability of Obstacles", min_value=0.05, max_value=0.5, value=0.25, step=0.05)
        
        def create_maze(rows, cols, obstacle_prob):
            default_start = (1, 1)
            default_goal = (rows - 2, cols - 2)
            maze = MazeEnv(n_rows=rows, n_cols=cols, start=default_start, goal=default_goal, obstacle_prob=obstacle_prob)
            return maze
        
        if st.button("Create Maze"):
            maze = create_maze(rows, cols, obstacle_prob)
            st.session_state.maze = maze
            
        if "maze" in st.session_state:
            st.divider()
            
            # st.write(st.session_state.maze.start)
            # st.write(st.session_state.maze.goal)
            # st.write(st.session_state.maze.n_rows)
            # st.write(st.session_state.maze.n_cols)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                start_row = st.number_input("Start Row", min_value=0, max_value=st.session_state.maze.n_rows-1, value=st.session_state.maze.start[0], key="start_row")
            with col2:
                start_col = st.number_input("Start Column", min_value=0, max_value=st.session_state.maze.n_cols-1, value=st.session_state.maze.start[1], key="start_col")
            with col3:
                goal_row = st.number_input("Goal Row", min_value=0, max_value=st.session_state.maze.n_rows-1, value=st.session_state.maze.goal[0], key="goal_row")
            with col4:
                goal_col = st.number_input("Goal Column", min_value=0, max_value=st.session_state.maze.n_cols-1, value=st.session_state.maze.goal[1], key="goal_col")
            
            if st.button("Customize Maze: Start & Goal"):
                st.session_state.maze.start = (int(start_row), int(start_col))
                st.session_state.maze.goal = (int(goal_row), int(goal_col))
                st.session_state.maze.maze[st.session_state.maze.start] = 0
                st.session_state.maze.maze[st.session_state.maze.goal] = 0
                
            st.plotly_chart(st.session_state.maze.render(), use_container_width=True)
            st.divider()
            
            # col1_1, col1_2, col1_3 = st.columns(3)
            # with col1_1:
            #     episodes = st.slider("Episodes", min_value=100, max_value=2000, value=500, step=100)
            # with col1_2:
            #     learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            # with col1_3:
            #     discount = st.slider("Discount Factor", min_value=0.5, max_value=1.0, value=0.95, step=0.01)
                
            # col2_1, col2_2, col2_3 = st.columns(3)
            # with col2_1:
            #     epsilon = st.slider("Initial Epsilon", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
            # with col2_2:
            #     epsilon_decay = st.slider("Epsilon Decay", min_value=0.90, max_value=1.0, value=0.995, step=0.001)
            # with col2_3:
            #     max_steps = st.slider("Max Steps per Episode", min_value=100, max_value=5000, value=1000, step=100)
            
            episodes = st.slider("Episodes", min_value=100, max_value=2000, value=500, step=100)
            learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            discount = st.slider("Discount Factor", min_value=0.5, max_value=1.0, value=0.95, step=0.01)
            epsilon = st.slider("Initial Epsilon", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
            epsilon_decay = st.slider("Epsilon Decay", min_value=0.90, max_value=1.0, value=0.995, step=0.001)
            max_steps = st.slider("Max Steps per Episode", min_value=100, max_value=5000, value=1000, step=100)
            
            if st.button("Train Agent"):
                with st.spinner("Training..."):
                    agent = QLearningAgent(st.session_state.maze,
                                        learning_rate=learning_rate,
                                        discount=discount,
                                        epsilon=epsilon,
                                        epsilon_decay=epsilon_decay)
                    rewards = agent.train(episodes=episodes, max_steps=max_steps)
                    optimal_path = agent.get_optimal_path(max_steps=max_steps)
                st.plotly_chart(st.session_state.maze.render(path=optimal_path), use_container_width=True)
    st.feedback("thumbs")
    mention(
            label="GitHub Repo: verneylmavt/q-learning-maze",
            icon="github",
            url="https://github.com/verneylmavt/q-learning-maze"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
        

if __name__ == "__main__":
    main()