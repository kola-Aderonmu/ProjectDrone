import pygame
import random
from heapq import heappop, heappush
import time

# Start Pygame
pygame.init()
pygame.mixer.init()

# Set up the window
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Sim Maze")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Load assets
try:
    drone_img = pygame.image.load("drone.jpg").convert_alpha()
    drone_img = pygame.transform.scale(drone_img, (20, 20))
except FileNotFoundError:
    print("drone.png not found, using circle instead")
    drone_img = None
try:
    crash_sound = pygame.mixer.Sound("crash.wav")
    win_sound = pygame.mixer.Sound("win_sound.wav")
except Exception as e:
    print(f"Sound load failed: {e}")
    crash_sound = None
    win_sound = None

# Fonts
title_font = pygame.font.SysFont(None, 72)
font = pygame.font.SysFont(None, 48)

# Global variables
drone_x = 50
drone_y = 350
drone_speed = 2
drone_radius = 10
goal_x = 700
goal_y = 100
goal_radius = 10
static_obstacles = [
    (100, 0, 40, 300),
    (100, 400, 40, 200),
    (250, 200, 40, 400),
    (400, 0, 40, 300),
    (400, 400, 40, 200),
    (0, 0, WIDTH, 10),
    (0, HEIGHT - 10, WIDTH, 10),
    (0, 0, 10, HEIGHT),
    (WIDTH - 10, 0, 10, HEIGHT)
]
enemy_drones = [
    [550, 200, 30, 30, 2, 0, 550, 200],
    [300, 350, 30, 30, 0, 0.5, drone_x, drone_y]
]
all_obstacles = []
path = []
path_index = 0
start_time = 0

# A* helpers
def heuristic(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def get_path(start, goal, obstacles):
    grid_size = 5
    start = (start[0] // grid_size, start[1] // grid_size)
    goal = (goal[0] // grid_size, goal[1] // grid_size)
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append((current[0] * grid_size, current[1] * grid_size))
                current = came_from[current]
            path.append((start[0] * grid_size, start[1] * grid_size))
            return path[::-1]

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            cost = grid_size if dx == 0 or dy == 0 else grid_size * 1.414
            tentative_g = g_score[current] + cost

            nx, ny = neighbor[0] * grid_size, neighbor[1] * grid_size
            if nx < 0 or nx >= WIDTH or ny < 0 or ny >= HEIGHT:
                continue
            rect = pygame.Rect(nx - drone_radius - 5, ny - drone_radius - 5, 
                             drone_radius * 2 + 10, drone_radius * 2 + 10)
            if any(rect.colliderect(pygame.Rect(obs[:4])) for obs in obstacles):
                continue

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))
    return []

def evade_path(start, enemy_pos, obstacles, max_steps=20):
    """Mini A* to find a short escape path away from enemy."""
    grid_size = 5
    start = (start[0] // grid_size, start[1] // grid_size)
    enemy = (enemy_pos[0] // grid_size, enemy_pos[1] // grid_size)
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, enemy)}

    steps = 0
    while open_set and steps < max_steps:
        current = heappop(open_set)[1]
        steps += 1

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            cost = grid_size if dx == 0 or dy == 0 else grid_size * 1.414
            tentative_g = g_score[current] + cost

            nx, ny = neighbor[0] * grid_size, neighbor[1] * grid_size
            if nx < 0 or nx >= WIDTH or ny < 0 or ny >= HEIGHT:
                continue
            rect = pygame.Rect(nx - drone_radius - 5, ny - drone_radius - 5, 
                             drone_radius * 2 + 10, drone_radius * 2 + 10)
            if any(rect.colliderect(pygame.Rect(obs[:4])) for obs in obstacles):
                continue

            dist_to_enemy = heuristic(neighbor, enemy)
            if dist_to_enemy < 10:  # Reduced to 50 pixels (10 grid units)
                continue

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g - dist_to_enemy  # Maximize distance
                heappush(open_set, (f_score[neighbor], neighbor))

    # Reconstruct path
    if open_set:
        current = heappop(open_set)[1]
        path = []
        while current in came_from:
            path.append((current[0] * grid_size, current[1] * grid_size))
            current = came_from[current]
        path.append((start[0] * grid_size, start[1] * grid_size))
        return path[::-1]
    return []

def reset_game():
    global drone_x, drone_y, path, path_index, enemy_drones, start_time, all_obstacles
    drone_x = 50
    drone_y = 350
    enemy_drones = [
        [550, 200, 30, 30, 2, 0, 550, 200],
        [300, 350, 30, 30, 0, 0.5, drone_x, drone_y]
    ]
    all_obstacles = static_obstacles + [obs[:4] for obs in enemy_drones]
    path = get_path((drone_x, drone_y), (goal_x, goal_y), all_obstacles)
    path_index = 0
    start_time = time.time()

def main():
    global drone_x, drone_y, path, path_index, enemy_drones, start_time, all_obstacles
    reset_game()
    if crash_sound:
        crash_sound.play()
        print("Crash sound should play now!")

    running = True
    clock = pygame.time.Clock()
    replan_timer = 0
    game_over = False
    win = False
    state = "title"
    evade_path_list = []
    evade_path_index = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if state == "title" and event.key == pygame.K_SPACE:
                    state = "game"
                    reset_game()
                if game_over and event.key == pygame.K_r:
                    state = "game"
                    game_over = False
                    win = False
                    reset_game()

        if state == "title":
            screen.fill(BLACK)
            title = title_font.render("Drone Maze", True, WHITE)
            start_text = font.render("Press SPACE to Start", True, WHITE)
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 50))
            screen.blit(start_text, (WIDTH // 2 - start_text.get_width() // 2, HEIGHT // 2 + 50))
            pygame.display.flip()

        elif state == "game" and not game_over:
            # Move enemy drones
            for enemy in enemy_drones:
                ex, ey, ew, eh, sx, sy, tx, ty = enemy
                if enemy == enemy_drones[0]:
                    ex += sx
                    if ex <= 550 or ex + ew >= 650:
                        enemy[4] = -sx
                else:
                    if ex < drone_x:
                        ex += abs(sx)
                    elif ex > drone_x:
                        ex -= abs(sx)
                    if ey < drone_y:
                        ey += abs(sy)
                    elif ey > drone_y:
                        ey -= abs(sy)
                if ex <= 10 or ex + ew >= WIDTH - 10:
                    enemy[4] = -enemy[4]
                if ey <= 10 or ey + eh >= HEIGHT - 10:
                    enemy[5] = -enemy[5]
                enemy[0], enemy[1] = ex, ey

            # Check enemy proximity
            evade = False
            nearest_enemy = None
            for enemy in enemy_drones[1:]:
                enemy_center_x = enemy[0] + enemy[2] / 2
                enemy_center_y = enemy[1] + enemy[3] / 2
                drone_center_x = drone_x
                drone_center_y = drone_y
                dist = ((drone_center_x - enemy_center_x) ** 2 + (drone_center_y - enemy_center_y) ** 2) ** 0.5
                if dist < 75:
                    evade = True
                    nearest_enemy = enemy
                    break

            # Replan path
            replan_timer += 1
            if replan_timer >= 5 or evade:  # Replan more often
                all_obstacles = static_obstacles + [obs[:4] for obs in enemy_drones]
                path = get_path((drone_x, drone_y), (goal_x, goal_y), all_obstacles)
                path_index = 0
                replan_timer = 0
                if not path:
                    print(f"No path found at ({drone_x:.1f}, {drone_y:.1f})")
                if evade:
                    evade_path_list = evade_path((drone_x, drone_y), (nearest_enemy[0], nearest_enemy[1]), all_obstacles)
                    evade_path_index = 0

            # Move drone
            moved = False
            if evade and nearest_enemy and evade_path_list and evade_path_index < len(evade_path_list):
                target_x, target_y = evade_path_list[evade_path_index]
                dx, dy = target_x - drone_x, target_y - drone_y
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist <= drone_speed:
                    drone_x, drone_y = target_x, target_y
                    evade_path_index += 1
                    moved = True
                else:
                    if dist > 0:
                        new_x = drone_x + drone_speed * dx / dist
                        new_y = drone_y + drone_speed * dy / dist
                        test_rect = pygame.Rect(new_x - drone_radius, new_y - drone_radius, 
                                              drone_radius * 2, drone_radius * 2)
                        if not any(test_rect.colliderect(pygame.Rect(obs[:4])) for obs in all_obstacles):
                            drone_x, drone_y = new_x, new_y
                            moved = True
            elif path and path_index < len(path):
                target_x, target_y = path[path_index]
                dx, dy = target_x - drone_x, target_y - drone_y
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist <= drone_speed:
                    drone_x, drone_y = target_x, target_y
                    path_index += 1
                    moved = True
                else:
                    if dist > 0:
                        new_x = drone_x + drone_speed * dx / dist
                        new_y = drone_y + drone_speed * dy / dist
                        test_rect = pygame.Rect(new_x - drone_radius, new_y - drone_radius, 
                                              drone_radius * 2, drone_radius * 2)
                        if not any(test_rect.colliderect(pygame.Rect(obs[:4])) for obs in all_obstacles):
                            drone_x, drone_y = new_x, new_y
                            moved = True
            elif not path:  # Fallback
                dx = goal_x - drone_x
                dy = goal_y - drone_y
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if nearest_enemy:
                    ex, ey = nearest_enemy[0], nearest_enemy[1]
                    edx = drone_x - ex
                    edy = drone_y - ey
                    e_dist = (edx ** 2 + edy ** 2) ** 0.5
                    if e_dist > 0:
                        dx += edx / e_dist * 50
                        dy += edy / e_dist * 50
                        dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist > 0:
                    new_x = drone_x + drone_speed * dx / dist
                    new_y = drone_y + drone_speed * dy / dist
                    test_rect = pygame.Rect(new_x - drone_radius, new_y - drone_radius, 
                                          drone_radius * 2, drone_radius * 2)
                    if not any(test_rect.colliderect(pygame.Rect(obs[:4])) for obs in all_obstacles):
                        drone_x, drone_y = new_x, new_y
                        moved = True

            if not moved:
                print(f"Drone stuck at ({drone_x:.1f}, {drone_y:.1f})")

            # Collision check with static obstacles
            drone_rect = pygame.Rect(drone_x - drone_radius, drone_y - drone_radius, 
                                   drone_radius * 2, drone_radius * 2)
            for obs in all_obstacles:
                obs_rect = pygame.Rect(obs[:4])
                if drone_rect.colliderect(obs_rect):
                    print(f"Collision at drone ({drone_x:.1f}, {drone_y:.1f}) with obstacle {obs}")
                    if crash_sound:
                        crash_sound.play()
                        time.sleep(0.5)
                    print("Drone crashed into obstacle!")
                    game_over = True
                    break

            # Collision check with enemy drones
            for enemy in enemy_drones:
                enemy_rect = pygame.Rect(enemy[0], enemy[1], enemy[2], enemy[3])
                if drone_rect.colliderect(enemy_rect):
                    print(f"Collision at drone ({drone_x:.1f}, {drone_y:.1f}) with enemy at ({enemy[0]:.1f}, {enemy[1]:.1f})")
                    if crash_sound:
                        crash_sound.play()
                        time.sleep(0.5)
                    print("Drone crashed into enemy!")
                    game_over = True
                    break

            # Goal check
            if abs(drone_x - goal_x) < drone_radius and abs(drone_y - goal_y) < drone_radius:
                if win_sound:
                    win_sound.play()
                    time.sleep(0.5)
                print(f"Goal reached! Time: {time.time() - start_time:.1f}s")
                game_over = True
                win = True

        # Draw
        screen.fill(BLACK)
        if state == "game":
            for obs in static_obstacles:
                pygame.draw.rect(screen, RED, obs)
            for enemy in enemy_drones:
                pygame.draw.rect(screen, YELLOW, enemy[:4])
            pygame.draw.circle(screen, GREEN, (goal_x, goal_y), goal_radius)
            for px, py in path:
                pygame.draw.circle(screen, BLUE, (px, py), 2)
            if drone_img:
                screen.blit(drone_img, (int(drone_x - drone_radius), int(drone_y - drone_radius)))
            else:
                pygame.draw.circle(screen, WHITE, (int(drone_x), int(drone_y)), drone_radius)

            # Score
            elapsed = time.time() - start_time
            score_text = font.render(f"Time: {elapsed:.1f}s", True, WHITE)
            screen.blit(score_text, (10, 10))

            if game_over:
                if win:
                    text = font.render("Mission Complete! (R to Restart)", True, GREEN)
                else:
                    text = font.render("Mission Failed! (R to Restart)", True, RED)
                screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()