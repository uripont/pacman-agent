# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='LaPulga', second='LaPulga', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class LaPulga(CaptureAgent):
    """**LaPulga v0.1.0**, by Jorge and Oriol
    
    CURRENT APPROACH:
    - Chooses "mode"/strategy: Attack, Defend, Return
    - Each mode has a single goal (position) selection strategy:
        - Attack: Closest food on opponent's side
        - Defend: Closest visible invader (enemy Pacman) on our side
        - Return: Closest reachable position on home boundary
    - Uses A* search to find optimal path to goal
        - Avoids walls and dangerous ghosts
        - Heuristic is Manhattan distance to goal
    """
    
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        #default mode is attack
        self.start = None
        self.mode = 'attack'
        self.goal = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        
    def choose_action(self, game_state):
        """
        Choose the next action based on the current strategy and goal.
        """
        #update the agent's strategy (Attack, Defend, Return) based on the current game state
        self.decide_strategy(game_state)
        
        #select a specific goal position (x, y) on the grid based on the chosen strategy
        self.goal = self.choose_goal(game_state)

        #use A* search to find the optimal path to the goal
        #the search considers walls and avoids dangerous ghosts
        if self.goal:
            path = self.a_star_search(game_state, self.goal)
            if path:
                #if a path is found, take the first step in that path
                return path[0]

            
        # TODO: Improve fallback mechanism, never should be reached
        #fallback mechanism
        #if no path is found or no goal is set, choose a random legal action
        #we try to avoid stopping or reversing if possible to keep moving
        actions = game_state.get_legal_actions(self.index)
        good_actions = [a for a in actions if a != Directions.STOP]
        
        if not good_actions:
            return Directions.STOP
            
        return random.choice(good_actions)

    # Information gathering helpers -----------------
    def get_carrying_food(self, game_state):
        """Get the amount of food currently being carried by this agent."""
        return game_state.get_agent_state(self.index).num_carrying

    def get_time_remaining(self, game_state):
        """Get remaining game time."""
        return game_state.data.timeleft

    def has_visible_invaders(self, game_state):
        """Check if there are visible enemy Pacmen (invaders) on our side."""
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return any(enemy.is_pacman and enemy.get_position() is not None for enemy in enemies)

    def is_on_own_side(self, game_state):
        """Check if this agent is on its own side (not a Pacman)."""
        return not game_state.get_agent_state(self.index).is_pacman

    def get_current_position(self, game_state):
        """Get the current position of this agent."""
        return game_state.get_agent_position(self.index)

    def get_food_list(self, game_state):
        """Get the list of food dots on the opponent's side."""
        food = self.get_food(game_state)
        return food.as_list()

    def get_visible_invaders(self, game_state):
        """Get list of visible enemy Pacmen (invaders) positions."""
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [enemy.get_position() for enemy in enemies 
                   if enemy.is_pacman and enemy.get_position() is not None]
        return invaders

    def get_home_boundary_x(self, game_state):
        """Get the x-coordinate of the home boundary (midpoint of the map)."""
        return (game_state.get_walls().width - 1) // 2
    
    
    # Strategy decision helpers -----------------
    def should_return(self, game_state):
        """Determine if the agent should return home with its food."""
        carrying = self.get_carrying_food(game_state)
        time_left = self.get_time_remaining(game_state)

        # Return if carrying a significant amount of food (e.g., >= 3)
        if carrying >= 3:
            return True

        # Return if carrying ANY food and time is running out (e.g., < 150 turns left)
        if carrying > 0 and time_left < 150:
            return True

        return False

    def should_defend(self, game_state):
        """Determine if the agent should switch to defense mode."""
        return self.has_visible_invaders(game_state) and self.is_on_own_side(game_state)

    
    def decide_strategy(self, game_state):
        """
        Determines the agent's mode (attack, defend, return) based on the current game state.
        """
        if self.should_return(game_state):
            self.mode = 'return'
        elif self.should_defend(game_state):
            self.mode = 'defend'
        else:
            self.mode = 'attack'
            

    # Goal selection helpers -----------------
    def get_closest_food(self, game_state):
        """Get the closest food dot to the current position."""
        current_pos = self.get_current_position(game_state)
        food_list = self.get_food_list(game_state)
        
        if not food_list:
            return None
        
        closest_food = min(food_list, key=lambda pos: self.get_maze_distance(current_pos, pos))
        return closest_food

    def get_closest_home_entry(self, game_state):
        """Get the closest reachable position on the home boundary."""
        current_pos = self.get_current_position(game_state)
        boundary_x = self.get_home_boundary_x(game_state)
        walls = game_state.get_walls()
        
        # Find all valid positions on the boundary
        valid_entries = []
        for y in range(walls.height):
            if not walls[boundary_x][y]:
                valid_entries.append((boundary_x, y))
        
        if not valid_entries:
            return None
        
        closest_entry = min(valid_entries, key=lambda pos: self.get_maze_distance(current_pos, pos))
        return closest_entry

    def get_closest_invader(self, game_state):
        """Get the closest visible invader to the current position."""
        current_pos = self.get_current_position(game_state)
        invaders = self.get_visible_invaders(game_state)
        
        if not invaders:
            return None
        
        closest_invader = min(invaders, key=lambda pos: self.get_maze_distance(current_pos, pos))
        return closest_invader

    def choose_goal(self, game_state):
        """
        Selects the goal position based on the current mode.
        """
        if self.mode == 'attack':
            goal = self.get_closest_food(game_state)
            return goal if goal else self.start

        elif self.mode == 'return':
            goal = self.get_closest_home_entry(game_state)
            return goal if goal else self.start

        elif self.mode == 'defend':
            goal = self.get_closest_invader(game_state)
            return goal if goal else self.start

        else: # TODO: should not happen
            return self.start


    def a_star_search(self, game_state, goal):
        """
        A* search from current position to goal.
        Avoids ghosts.
        """
        start_pos = game_state.get_agent_position(self.index)
        
        # Priority Queue: (priority, (position, path))
        pq = util.PriorityQueue()
        pq.push((start_pos, []), 0)
        
        visited = set()
        
        # Get dangerous positions (ghosts)
        dangers = []
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        for enemy in enemies:
            if not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer <= 0:
                # Dangerous ghost
                ghost_pos = enemy.get_position()
                dangers.append(ghost_pos)
                # Add adjacent positions to dangers for safety buffer
                gx, gy = ghost_pos
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    dangers.append((gx+dx, gy+dy))
        
        while not pq.is_empty():
            curr_pos, path = pq.pop()
            
            if curr_pos in visited:
                continue
            visited.add(curr_pos)
            
            if curr_pos == goal:
                return path
            
            # Limit path length to avoid timeout
            if len(path) > 20: 
                continue

            x, y = curr_pos
            
            # Neighbors
            candidates = [
                (0, 1, Directions.NORTH),
                (0, -1, Directions.SOUTH),
                (1, 0, Directions.EAST),
                (-1, 0, Directions.WEST)
            ]
            
            for dx, dy, action in candidates:
                next_x, next_y = int(x + dx), int(y + dy)
                next_pos = (next_x, next_y)
                
                # Check walls
                if game_state.get_walls()[next_x][next_y]:
                    continue
                
                # Check danger
                if next_pos in dangers:
                    continue
                
                # Heuristic
                g = len(path) + 1
                h = self.get_maze_distance(next_pos, goal) # Manhattan distance heuristic
                f = g + h
                
                new_path = path + [action]
                pq.push((next_pos, new_path), f)
                
        return None
