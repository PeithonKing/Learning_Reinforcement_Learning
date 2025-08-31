import gymnasium as gym
import pygame

# Initialize environment
env = gym.make("MountainCar-v0", render_mode="human")

# Initialize pygame
pygame.init()
window = pygame.display.set_mode((200, 200))  # dummy window for event handling
pygame.display.set_caption("MountainCar Controller")

clock = pygame.time.Clock()

while True:
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = 1  # default: no push

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                raise SystemExit

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 2

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        clock.tick(30)  # Limit to 30 FPS

    print(f"Episode finished! Total reward: {total_reward:.2f}")
