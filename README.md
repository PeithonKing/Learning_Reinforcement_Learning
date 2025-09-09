# Learning Reinforcement Learning

This repository documents my journey in learning reinforcement learning, featuring implementations of algorithms like DQN and DDPG, as well as custom Gymnasium environments for experimentation.

## Progress

Here is a description (in chronological order) which folder (project) does what along with a short description. However a detailed description is provided in each folder.

- **[snake_ladder/](snake_ladder/)**: Using Q table. Very basic, numpy implementation. The game is, if the player is given a magical power of precisely controlling the outcome of the dice, what is the minimum number of steps the player can complete the game in.
- **[mountain_car/](mountain_car/)**: Learned Deep Q learning. Used a basic neural network (using pytorch) to train the [mountain car environment](https://gymnasium.farama.org/environments/classic_control/mountain_car/) from [gymnasium](https://gymnasium.farama.org/).
- **[gym_envs/](gym_envs/)**: Collection of custom Gymnasium environments. All custom gym envs we build go here, and we would use these to train RL.
- **[line_follower_v0/](line_follower_v0/)**: Using the custom env at [gym_envs/line_follower_v0](gym_envs/line_follower_v0). We train another DQN model. This is the second DQN model we trained.
- **[line_follower_v1/](line_follower_v1/)**: Using the custom env at [gym_envs/line_follower_v1](gym_envs/line_follower_v1). But this time, the actoion space was cotinuous, so we needed to learn actor-critic algorithm for this and using that instead of DQN.

## Plans

- [x] snake ladder
- [x] mountain_car
- [x] line_follower_v0
- [x] line_follower_v1
- [ ] badminton
- [ ] flappy birds
- [ ] fleet battle
- [ ] minesweeper

## Installation

1. **Create a virtual environment** (optional but highly recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install PyTorch**:
   - GPU is mostly not needed, and all code works fast enough on CPU. You can install PyTorch for CPU like this:

     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     ```

   - however if you absolutely want to use GPU and have a cuda supported NVIDIA Card, follow the instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

3. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the custom environments**:

   ```bash
   cd gym_envs
   pip install -e .  # install the custom gym environments
   cd ..  # go back to the root directory
   ```
