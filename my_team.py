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
# Agent #
##########

class LaPulga(CaptureAgent):
    """LaPulga v0.2.2, by Jorge and Oriol
    
    MAIN FIX: Fixed return home goal calculation. Previously got stuck on the boundary,
    now correctly goes to the left/right side depending on team color (tested in blue team).
    
    CURRENT APPROACH: Four-layer decision system:
    1. Situation: Detects game state (position, food, threats, etc.)
    2. Strategy: Maps situations to modes (Attack, Defend, Return, Hunt scared)
       - Supports patches for specific plays/tuning
    3. Goal: Chooses target position based on strategy
    4. Pathfinding: A* to find optimal path to goal
       - Avoids walls and dangerous ghosts
       - Heuristic is Manhattan distance
    """
    
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        
        # State vars
        self.start = None
        self.situation = None
        self.strategy = None
        self.goal = None
        
        # Initialize layers
        self.situation_analyzer = SituationAnalyzer()
        self.strategy_selector = StrategySelector()
        self.goal_chooser = GoalChooser()
        self.path_finder = PathFinder()

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        
    # The high-level decision algorithm
    def choose_action(self, game_state):
        self.situation = self.situation_analyzer.analyze(game_state, self)
        self.strategy = self.strategy_selector.select_strategy(self.situation)
        self.goal = self.goal_chooser.choose_goal(game_state, self, self.situation, self.strategy)

        if self.goal:
            path = self.path_finder.search(game_state, self, self.goal)
            if path:
                return path[0]
        
        # Fallback: return STOP to make issues visible
        # TODO: NEVER should use fallback, only for debugging
        return Directions.STOP


###########################
# Layer 1: Situation
###########################

class SituationAnalyzer:
    def analyze(self, game_state, agent):
        situation = Situation()
        
        # Basic position info
        agent_pos = game_state.get_agent_position(agent.index)
        situation.agent_pos = agent_pos
        situation.on_own_side = not game_state.get_agent_state(agent.index).is_pacman
        
        # Court halves
        walls = game_state.get_walls()
        midline_x = (walls.width - 1) // 2
        situation.court_half = 'own' if agent_pos[0] < midline_x else 'opponent'
        
        # Upper/lower half (y-axis)
        midline_y = walls.height // 2
        situation.vertical_half = 'upper' if agent_pos[1] >= midline_y else 'lower'
        
        # Player counts
        enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
        teammates = [game_state.get_agent_state(i) for i in agent.get_team(game_state) if i != agent.index]
        
        situation.enemies_alive = sum(1 for e in enemies if e.get_position() is not None)
        situation.teammates_alive = sum(1 for t in teammates if t.get_position() is not None)
        
        # Spawn zone check (near start position)
        situation.in_spawn_zone = agent.get_maze_distance(agent_pos, agent.start) < 5
        
        # Food information
        food_list = self._get_food_list(game_state, agent)
        situation.food_remaining = len(food_list)
        
        if food_list:
            closest_food_dist = min(agent.get_maze_distance(agent_pos, food) for food in food_list)
            situation.closest_food_distance = closest_food_dist
            closest_food = min(food_list, key=lambda f: agent.get_maze_distance(agent_pos, f))
            situation.closest_food_pos = closest_food
        else:
            situation.closest_food_distance = float('inf')
            situation.closest_food_pos = None
        
        # Enemy proximity
        situation.visible_invaders = self._get_visible_invaders(game_state, agent)
        situation.has_invaders_visible = len(situation.visible_invaders) > 0
        
        if situation.visible_invaders:
            closest_invader_dist = min(agent.get_maze_distance(agent_pos, inv) for inv in situation.visible_invaders)
            situation.closest_invader_distance = closest_invader_dist
        else:
            situation.closest_invader_distance = float('inf')
        
        # Pellet information (power pellets)
        situation.capsules = agent.get_capsules(game_state)
        situation.has_capsules = len(situation.capsules) > 0
        
        if situation.capsules:
            closest_capsule_dist = min(agent.get_maze_distance(agent_pos, cap) for cap in situation.capsules)
            situation.closest_capsule_distance = closest_capsule_dist
        else:
            situation.closest_capsule_distance = float('inf')
        
        # Carrying food info
        situation.carrying_food = self._get_carrying_food(game_state, agent)
        
        # Time remaining
        situation.time_remaining = self._get_time_remaining(game_state)
        
        # Scared ghosts (enemies under pellet effect)
        situation.scared_ghosts = [e for e in enemies 
                                   if not e.is_pacman and e.get_position() is not None and e.scared_timer > 0]
        situation.has_scared_ghosts = len(situation.scared_ghosts) > 0
        
        return situation
    
    # Helpers (that we have from v0.1.0) -----
    def _get_carrying_food(self, game_state, agent):
        """Get the amount of food currently being carried by this agent."""
        return game_state.get_agent_state(agent.index).num_carrying

    def _get_time_remaining(self, game_state):
        """Get remaining game time."""
        return game_state.data.timeleft

    def _get_food_list(self, game_state, agent):
        """Get the list of food dots on the opponent's side."""
        food = agent.get_food(game_state)
        return food.as_list()

    def _get_visible_invaders(self, game_state, agent):
        """Get list of visible enemy Pacman (invaders) positions."""
        enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
        invaders = [enemy.get_position() for enemy in enemies 
                   if enemy.is_pacman and enemy.get_position() is not None]
        return invaders


class Situation:
    def __init__(self):
        self.agent_pos = None
        self.on_own_side = True
        self.court_half = 'own'  # 'own' or 'opponent'
        self.vertical_half = 'lower'  # 'upper' or 'lower'
        
        self.enemies_alive = 0
        self.teammates_alive = 0
        self.in_spawn_zone = False
        
        self.food_remaining = 0
        self.closest_food_distance = float('inf')
        self.closest_food_pos = None
        
        self.visible_invaders = []
        self.has_invaders_visible = False
        self.closest_invader_distance = float('inf')
        
        self.capsules = []
        self.has_capsules = False
        self.closest_capsule_distance = float('inf')
        
        self.carrying_food = 0
        self.time_remaining = 0
        
        self.scared_ghosts = []
        self.has_scared_ghosts = False


###########################
# Layer 2: Strategy
###########################

class StrategySelector:
    """
    Maps situations to strategies, with support for patches.
    A "patch" is a override for certain "rehearsed plays" detected in the situation.
    """
    
    def __init__(self):
        self.patches = {}
    
    def add_patch(self, patch_name, patch_condition_fn, patch_strategy_fn):
        """
        Add a patch that overrides strategy selection for specific situations.
        
        Args:
            patch_name: Unique identifier for this patch
            patch_condition_fn: callable(situation) -> bool
            patch_strategy_fn: callable(situation) -> str (strategy name)
        """
        self.patches[patch_name] = (patch_condition_fn, patch_strategy_fn)
    
    def select_strategy(self, situation):
        """
        Select strategy for this situation, checking patches first.
        
        Returns:
            str: strategy name ('attack', 'defend', 'return', 'scout', etc.)
        """
        # Check patches first (specific play tuning)
        for patch_name, (condition_fn, strategy_fn) in self.patches.items():
            if condition_fn(situation):
                return strategy_fn(situation)
        
        # Default strategy selection logic, draft for now
        return self._default_strategy(situation)
    
    def _default_strategy(self, situation):
        """
        Default strategy selection based on situation.
        This is the general behavior that patches improve upon.
        """
        # If there are visible invaders and we're on our side, defend
        if situation.has_invaders_visible and situation.on_own_side:
            return 'defend'
        
        #If carrying food and either:
        #   - carrying >= 3 food, or
        #   - carrying > 0 and time is running out (< 150 turns)
        if situation.carrying_food >= 3:
            return 'return'
        
        if situation.carrying_food > 0 and situation.time_remaining < 150:
            return 'return'
        
        # Scared ghosts on opponent side? Go attack them
        if situation.has_scared_ghosts and not situation.on_own_side:
            return 'hunt_scared'
        
        # Default attack
        return 'attack'


###########################
# Layer 3: Goal
###########################

class GoalChooser:
    """
    Chooses the immediate next-step goal position based on strategy.
    The goal is NOT necessarily the final destination, it's the position to fulfill the next strategic step.
    """
    
    def choose_goal(self, game_state, agent, situation, strategy):
        """
        Choose the goal position for the current turn.
        
        Returns:
            (x, y) position to move toward
        """
        if strategy == 'attack':
            return self._goal_attack(game_state, agent, situation)
        
        elif strategy == 'defend':
            return self._goal_defend(game_state, agent, situation)
        
        elif strategy == 'return':
            return self._goal_return(game_state, agent, situation)
        
        elif strategy == 'hunt_scared':
            return self._goal_hunt_scared(game_state, agent, situation)
        
        elif strategy == 'scout':
            return self._goal_scout(game_state, agent, situation)
        
        else:
            # Default fallback
            return agent.start
    
    # Drafting goal strategies
    def _goal_attack(self, game_state, agent, situation):
        """Attack strategy: Go for closest food."""
        if situation.closest_food_pos:
            return situation.closest_food_pos
        return agent.start
    
    def _goal_defend(self, game_state, agent, situation):
        """Defend strategy: Go for closest visible invader."""
        if situation.visible_invaders:
            closest_inv = min(situation.visible_invaders, 
                            key=lambda inv: agent.get_maze_distance(situation.agent_pos, inv))
            return closest_inv
        return agent.start
    
    def _goal_return(self, game_state, agent, situation):
        """Return strategy: Go to closest home entry point."""
        walls = game_state.get_walls()
        
        # We previously incorrectly divided walls.width - 1 by 2, getting stuck in the boundary
        # Red team: home is left side (x=0 area)
        # Blue team: home is right side (x=walls.width-1 area)
        if game_state.is_on_red_team(agent.index):
            # Red team returns to left side
            boundary_x = 0
        else:
            # Blue team returns to right side
            boundary_x = walls.width - 1
        
        valid_entries = []
        for y in range(walls.height):
            if not walls[boundary_x][y]:
                valid_entries.append((boundary_x, y))
        
        if valid_entries:
            return min(valid_entries, 
                      key=lambda pos: agent.get_maze_distance(situation.agent_pos, pos))
        
        return agent.start
    
    def _goal_hunt_scared(self, game_state, agent, situation):
        """Hunt scared ghosts: Go for closest scared ghost."""
        if situation.scared_ghosts:
            closest_ghost_pos = min(
                [ghost.get_position() for ghost in situation.scared_ghosts],
                key=lambda pos: agent.get_maze_distance(situation.agent_pos, pos)
            )
            return closest_ghost_pos
        
        # Fallback to attack if no scared ghosts visible
        return self._goal_attack(game_state, agent, situation)
    
    def _goal_scout(self, game_state, agent, situation):
        """Scout strategy: Move toward food to explore the map."""
        if situation.closest_food_pos:
            return situation.closest_food_pos
        return agent.start


###########################
# Layer 4: Pathfinding (A*)
###########################

class PathFinder:
    """
    A* pathfinding from current position to goal.
    Avoids walls and dangerous ghosts.
    Heuristic currently is Manhattan distance to goal.
    """
    
    def search(self, game_state, agent, goal):
        """
        Find optimal path to goal using A* search.
        
        Args:
            game_state: Current game state
            agent: The agent searching (needs get_maze_distance, get_opponents)
            goal: Target position (x, y)
        
        Returns:
            List of actions to reach goal, or None if no path found
        """
        start_pos = game_state.get_agent_position(agent.index)
        
        # Priority Queue: (priority, (position, path))
        pq = util.PriorityQueue()
        pq.push((start_pos, []), 0)
        
        visited = set()
        
        # Identify dangerous positions (non-pacman ghosts)
        dangers = self._get_danger_positions(game_state, agent)
        
        while not pq.is_empty():
            curr_pos, path = pq.pop()
            
            if curr_pos in visited:
                continue
            visited.add(curr_pos)
            
            if curr_pos == goal:
                return path
            
            # Limit path length to avoid timeout
            if len(path) > 200: # 20 was too short to go outside spawn
                continue

            x, y = curr_pos
            
            # Explore neighbors
            candidates = [
                (0, 1, Directions.NORTH),
                (0, -1, Directions.SOUTH),
                (1, 0, Directions.EAST),
                (-1, 0, Directions.WEST)
            ]
            
            for dx, dy, action in candidates:
                next_x, next_y = int(x + dx), int(y + dy)
                next_pos = (next_x, next_y)
                
                # Skip if wall
                if game_state.get_walls()[next_x][next_y]:
                    continue
                
                # Skip if dangerous position
                if next_pos in dangers:
                    continue
                
                # Calculate f = g + h (cost + heuristic)
                g = len(path) + 1
                h = agent.get_maze_distance(next_pos, goal)
                f = g + h
                
                new_path = path + [action]
                pq.push((next_pos, new_path), f)
                
        return None
    
    def _get_danger_positions(self, game_state, agent):
        """
        Get set of dangerous positions (non-pacman ghosts only).
        Only avoid ghosts that are very close.
        
        Returns:
            set of (x, y) positions to avoid
        """
        dangers = set()
        enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
        agent_pos = game_state.get_agent_position(agent.index)
        
        for enemy in enemies:
            # Only dangerous if: ghost (not pacman), visible, not scared and close
            if not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer <= 0:
                ghost_pos = enemy.get_position()
                # Hotfix:Only avoid if within 2 steps
                dist = agent.get_maze_distance(agent_pos, ghost_pos)
                if dist <= 2:
                    dangers.add(ghost_pos)
        
        return dangers
