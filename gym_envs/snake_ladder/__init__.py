from gymnasium.envs.registration import register

register(
    id="my_gym_envs/snake_ladder_v0",
    entry_point="snake_ladder.envs:SnakeLadderEnv",
)
