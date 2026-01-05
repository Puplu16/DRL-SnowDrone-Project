import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800, 600
STATE_SIZE = 6
ACTION_SIZE = 4
GAMMA = 0.99
LR = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
SNOWFLAKES = 120

# ---------------- DQN MODEL ----------------
class DQN(nn.Module):
    def __init__(self):  # Corrected double underscores
        super().__init__() # Corrected double underscores
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- CUSTOM ENVIRONMENT ----------------
class SnowDroneEnv:
    def __init__(self):  # Corrected double underscores
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snow Drone RL - Rescue Mission")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 28, bold=True)
        self.msg_font = pygame.font.SysFont("Arial", 50, bold=True)
        
        # Scenery setup
        self.snow_particles = [[random.randint(0, WIDTH), random.randint(0, HEIGHT)] for _ in range(SNOWFLAKES)]
        self.mtn_pts = [
            [( -100, 280), (150, 60), (400, 280)],
            [(200, 280), (450, 100), (750, 280)],
            [(550, 280), (800, 50), (1100, 280)]
        ]
        self.tree_positions = [random.randint(0, WIDTH) for _ in range(15)]
        self.reset()

    def reset(self):
        self.drone = np.array([WIDTH//2, HEIGHT-70], dtype=np.float32)
        self.battery = 100.0
        self.score = 0
        self.steps = 0
        self.items = [np.array([random.randint(50, WIDTH-50), random.randint(300, HEIGHT-80)]) for _ in range(5)]
        self.obstacles = [np.array([random.randint(50, WIDTH-50), random.randint(300, HEIGHT-150)]) for _ in range(4)]
        return self.get_state()

    def get_state(self):
        if self.items:
            closest = min(self.items, key=lambda i: np.linalg.norm(self.drone - i))
            dist = np.linalg.norm(self.drone - closest)
        else:
            closest, dist = self.drone, 0
        
        return np.array([
            self.drone[0]/WIDTH, self.drone[1]/HEIGHT,
            closest[0]/WIDTH, closest[1]/HEIGHT,
            self.battery/100, dist/800
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = -0.01 
        speed = 8

        if self.items:
            old_dist = min(np.linalg.norm(self.drone - i) for i in self.items)
        else:
            old_dist = 0

        if action == 0: self.drone[1] -= speed # Up
        elif action == 1: self.drone[1] += speed # Down
        elif action == 2: self.drone[0] -= speed # Left
        elif action == 3: self.drone[0] += speed # Right

        self.drone = np.clip(self.drone, [0, 0], [WIDTH, HEIGHT])
        drain = 0.25 if self.drone[1] < 280 else 0.1
        self.battery -= drain

        if self.items:
            new_dist = min(np.linalg.norm(self.drone - i) for i in self.items)
            reward += 0.5 if new_dist < old_dist else -0.2

        new_items = []
        for item in self.items:
            if np.linalg.norm(self.drone - item) < 22:
                reward += 50
                self.score += 1
            else:
                new_items.append(item)
        self.items = new_items

        for obs in self.obstacles:
            if np.linalg.norm(self.drone - obs) < 25:
                reward -= 10
                self.battery -= 5

        done = False
        status = ""
        if self.battery <= 0:
            done = True
            reward -= 20
            status = "BATTERY EMPTY"
        elif self.steps > 1000:
            done = True
            status = "TIMEOUT"
        elif not self.items:
            done = True
            reward += 100
            status = "MISSION COMPLETE"

        return self.get_state(), reward, done, status

    def render(self, message=""):
        self.screen.fill((185, 205, 235)) 
        for pts in self.mtn_pts:
            pygame.draw.polygon(self.screen, (140, 160, 190), pts)
            tip = pts[1]
            cap = [tip, (tip[0]-45, tip[1]+45), (tip[0]+45, tip[1]+45)]
            pygame.draw.polygon(self.screen, (255, 255, 255), cap)

        pygame.draw.rect(self.screen, (240, 245, 255), (0, 280, WIDTH, HEIGHT-280))

        for x in self.tree_positions:
            ty = 285
            pygame.draw.rect(self.screen, (60, 35, 15), (x, ty-10, 6, 15))
            pygame.draw.polygon(self.screen, (25, 70, 35), [(x-18, ty-5), (x+24, ty-5), (x+3, ty-40)])

        for s in self.snow_particles:
            pygame.draw.circle(self.screen, (255, 255, 255), s, 2)
            s[1] += 2
            if s[1] > HEIGHT: 
                s[1] = 0
                s[0] = random.randint(0, WIDTH)

        for o in self.obstacles:
            pygame.draw.rect(self.screen, (110, 70, 40), (o[0]-12, o[1]-12, 24, 24))
        for i in self.items:
            pygame.draw.circle(self.screen, (200, 30, 30), i.astype(int), 10)
        
        pygame.draw.circle(self.screen, (40, 190, 40), self.drone.astype(int), 14)

        pygame.draw.rect(self.screen, (0, 255, 0), (20, 20, int(self.battery * 1.5), 25))
        score_surf = self.font.render(f"Score:{self.score}", True, (0, 0, 0))
        self.screen.blit(score_surf, (20, 55))

        if message:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0,0))
            msg_surf = self.msg_font.render(message, True, (255, 255, 255))
            self.screen.blit(msg_surf, (WIDTH//2 - 150, HEIGHT//2 - 25))

        pygame.display.flip()
        self.clock.tick(60)

# ---------------- AGENT LOGIC ----------------
class Agent:
    def __init__(self):  # Corrected double underscores
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.epsilon = 1.0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE-1)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            return torch.argmax(self.model(state_t)).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE: return
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        ns = torch.tensor(np.array(ns), dtype=torch.float32)
        a = torch.tensor(a).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        curr_q = self.model(s).gather(1, a).squeeze()
        next_q = self.target_model(ns).max(1)[0]
        target = r + (GAMMA * next_q * (1 - d))

        loss = nn.MSELoss()(curr_q, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(0.05, self.epsilon * 0.997)

# ---------------- MAIN ----------------
def main():
    env = SnowDroneEnv()
    agent = Agent()
    episodes = 200

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return

            action = agent.act(state)
            next_state, reward, done, status = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            
            agent.train()
            state = next_state
            total_reward += reward

            if ep % 3 == 0: 
                env.render()
        
        if ep % 5 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            env.render(status)
            pygame.time.delay(500)
            
        print(f"Episode {ep+1} | Score: {env.score} | Reward: {int(total_reward)} | Epsilon: {agent.epsilon:.2f}")

    torch.save(agent.model.state_dict(), "drone_pilot.pth")
    pygame.quit()

if __name__ == "__main__":  # Corrected double underscores
    main()