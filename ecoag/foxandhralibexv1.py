# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:52:14 2025

@author: mikeg
"""

import raylibpy as rl
import random
import math

# Game settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FOX_SPEED = 4
HOUND_SPEED = 0.06125
NUM_HOUNDS = 2
FOX_RADIUS = 10
HOUND_RADIUS = 10

# Helper function
def move_toward(target, current, speed):
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return current
    return (current[0] + 0.0001*speed * dx / dist, current[1] + speed * dy / dist)

# Initialize game
rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, b"Fox and Hounds")
rl.set_target_fps(60)

fox_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]

hounds = [[random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)] for _ in range(NUM_HOUNDS)]
h=[0,0]
i=0
for hh in hounds:
    h[0]=hh[0]
    h[1]=hh[1]
game_over = False

while not rl.window_should_close():
    hound_control_mode = False
    selected_hound_index = None

    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)
    
        rl.draw_text("Fox and Hounds", 10, 10, 20, rl.DARKGRAY)
    
        # Toggle control mode
        if rl.is_key_pressed(rl.KEY_C):
            hound_control_mode = not hound_control_mode
            selected_hound_index = None
    
        mouse_pos = rl.get_mouse_position()
    
        # Select hound under mouse
        if hound_control_mode:
            for i, h in enumerate(hounds):
                if rl.check_collision_point_circle(mouse_pos, rl.Vector2(h[0], h[1]), HOUND_RADIUS):
                    selected_hound_index = i
                    break
    
        # Update fox movement
        if not game_over and not hound_control_mode:
            if rl.is_key_down(rl.KEY_RIGHT): fox_pos[0] += FOX_SPEED
            if rl.is_key_down(rl.KEY_LEFT):  fox_pos[0] -= FOX_SPEED
            if rl.is_key_down(rl.KEY_UP):    fox_pos[1] -= FOX_SPEED
            if rl.is_key_down(rl.KEY_DOWN):  fox_pos[1] += FOX_SPEED
    
        # Update selected hound movement
        if hound_control_mode and selected_hound_index is not None:
            h = hounds[selected_hound_index]
            h=list(h)
            if rl.is_key_down(rl.KEY_RIGHT): h[0] += HOUND_SPEED
            if rl.is_key_down(rl.KEY_LEFT):  h[0] -= HOUND_SPEED
            if rl.is_key_down(rl.KEY_UP):    h[1] -= HOUND_SPEED
            if rl.is_key_down(rl.KEY_DOWN):  h[1] += HOUND_SPEED
            hounds[selected_hound_index] = tuple(h)
    
        # Update hounds (AI)
        if not game_over and not hound_control_mode:
            for i in range(NUM_HOUNDS):
                hounds[i] = move_toward(fox_pos, hounds[i], h[i])
                if math.hypot(hounds[i][0] - fox_pos[0], hounds[i][1] - fox_pos[1]) < FOX_RADIUS + HOUND_RADIUS:
                    game_over = True
    
        # Draw fox
        rl.draw_circle(int(fox_pos[0]), int(fox_pos[1]), FOX_RADIUS, rl.ORANGE)
    
        # Draw hounds
        for i, h in enumerate(hounds):
            color = rl.DARKGREEN
            if hound_control_mode and i == selected_hound_index:
                color = rl.RED
            rl.draw_circle(int(h[0]), int(h[1]), HOUND_RADIUS, color)
    
        # Game over message
        if game_over:
            rl.draw_text("Game Over! Fox was caught!", SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2, 20, rl.RED)
    
        # Control mode message
        if hound_control_mode:
            rl.draw_text("Hound Control Mode (Press C to exit)", 10, 40, 20, rl.BLUE)
    
        rl.end_drawing()

rl.close_window()