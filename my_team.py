# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions

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
    """LaPulga v0.4.1, by Jorge and Oriol
    
    MAIN FIX: Removed all fields and imports that were not used in this latest version.
    
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
        
        # Upper/lower half (y-axis)
        walls = game_state.get_walls()
        midline_y = walls.height // 2
        
        # Teammate position and vertical split
        teammates = [game_state.get_agent_state(i) for i in agent.get_team(game_state) if i != agent.index]
        if teammates and teammates[0].get_position() is not None:
            teammate_pos = teammates[0].get_position()
            # Determine which agent should focus on upper vs lower
            situation.should_focus_upper = agent_pos[1] >= teammate_pos[1]
        else:
            situation.should_focus_upper = None
        
        # Player counts and enemies list
        enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
        
        # Food information
        food_list = self._get_food_list(game_state, agent)
        situation.food_remaining = len(food_list)
        
        if food_list:
            closest_food = min(food_list, key=lambda f: agent.get_maze_distance(agent_pos, f))
            situation.closest_food_pos = closest_food
        else:
            situation.closest_food_pos = None
        
        # Enemy proximity
        situation.visible_invaders = self._get_visible_invaders(game_state, agent)
        situation.has_invaders_visible = len(situation.visible_invaders) > 0
        
        # Carrying food info
        situation.carrying_food = self._get_carrying_food(game_state, agent)
        
        # Time remaining
        situation.time_remaining = self._get_time_remaining(game_state)
        
        # Scared ghosts (enemies under pellet effect)
        situation.scared_ghosts = [e for e in enemies 
                                   if not e.is_pacman and e.get_position() is not None and e.scared_timer > 0]
        situation.has_scared_ghosts = len(situation.scared_ghosts) > 0
        
        # Dangerous ghosts (non-pacman, not scared, visible) - these can kill us when we're attacking
        situation.dangerous_ghosts = [e for e in enemies 
                                      if not e.is_pacman and e.get_position() is not None and e.scared_timer <= 0]
        situation.has_dangerous_ghosts = len(situation.dangerous_ghosts) > 0
        
        if situation.dangerous_ghosts:
            ghost_positions = [g.get_position() for g in situation.dangerous_ghosts]
            closest_ghost_dist = min(agent.get_maze_distance(agent_pos, gpos) for gpos in ghost_positions)
            situation.closest_dangerous_ghost_distance = closest_ghost_dist
            situation.closest_dangerous_ghost_pos = min(ghost_positions, 
                                                        key=lambda gpos: agent.get_maze_distance(agent_pos, gpos))
        else:
            situation.closest_dangerous_ghost_distance = float('inf')
            situation.closest_dangerous_ghost_pos = None
        
        # Our scared state (we are a pacman and have scared timer)
        agent_state = game_state.get_agent_state(agent.index)
        situation.we_are_scared = agent_state.is_pacman and agent_state.scared_timer > 0
        
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
        
        self.teammate_pos = None
        self.teammate_vertical_half = None
        self.should_focus_upper = None  # True if this agent should focus on upper half
        
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
        
        self.dangerous_ghosts = []
        self.has_dangerous_ghosts = False
        self.closest_dangerous_ghost_distance = float('inf')
        self.closest_dangerous_ghost_pos = None
        
        self.we_are_scared = False
        self.scared_timer_remaining = 0


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
            str: strategy name ('attack', 'defend', 'return', etc.)
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
        # If we are scared and would be in defend mode, retreat to enemy base instead
        if situation.we_are_scared and situation.has_invaders_visible and situation.on_own_side:
            return 'scare_retreat'
        
        # If there are visible invaders and we're on our side, defend
        if situation.has_invaders_visible and situation.on_own_side:
            return 'defend'
        
        #If carrying food and either:
        #   - carrying >= 3 food, or
        #   - carrying > 0 and time is running out (< 150 turns)
        if situation.carrying_food >= 5:
            return 'return'
        
        if situation.carrying_food > 0 and situation.time_remaining < 150:
            return 'return'
        
        # Always: when food is 2, we can return: picking more food is risky and does not bring reward
        # (game ends when all food except 2 at most is eaten)
        if situation.carrying_food > 0 and situation.food_remaining <= 2:
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
        
        elif strategy == 'scare_retreat':
            return self._goal_scare_retreat(game_state, agent, situation)
        
        else:
            # Default fallback
            return agent.start
    
    # Drafting goal strategies
    def _goal_attack(self, game_state, agent, situation):
        """Attack strategy: Go for closest food, preferring safe food when ghost is nearby."""
        if not situation.closest_food_pos:
            return agent.start
        
        food_list = agent.get_food(game_state).as_list()
        
        # If we have teammate position info, divide territory to avoid conflicts
        if situation.should_focus_upper is not None and len(food_list) > 1:
            # Split food into upper and lower halves
            walls = game_state.get_walls()
            midline_y = walls.height // 2
            
            # Partition food based on vertical position
            if situation.should_focus_upper:
                # This agent focuses on upper half
                assigned_food = [f for f in food_list if f[1] >= midline_y]
            else:
                # This agent focuses on lower half
                assigned_food = [f for f in food_list if f[1] < midline_y]
            
            # If we have food in our assigned territory, use that list
            if assigned_food:
                food_list = assigned_food
        
        # SAFETY CHECK: When a dangerous ghost is nearby, filter out unsafe food
        ghost_nearby_threshold = 6  # Distance at which we start being cautious
        
        if situation.has_dangerous_ghosts and situation.closest_dangerous_ghost_distance <= ghost_nearby_threshold:
            ghost_pos = situation.closest_dangerous_ghost_pos
            
            # Filter food to only safe options
            safe_food = [f for f in food_list 
                        if self._is_food_safe(game_state, agent, f, 
                                              situation.agent_pos, ghost_pos)]
            
            if safe_food:
                # Pick closest safe food
                return min(safe_food, key=lambda f: agent.get_maze_distance(situation.agent_pos, f))
            else:
                # No safe food! Switch to return mode or find escape route
                # Return to home boundary for safety
                return self._goal_return(game_state, agent, situation)
        
        # No immediate danger, go for closest food
        return min(food_list, key=lambda f: agent.get_maze_distance(situation.agent_pos, f))
    
    def _is_food_safe(self, game_state, agent, food_pos, agent_pos, ghost_pos):
        """Check if pursuing food is safe given a nearby ghost."""
        corridor_positions = PathFinder.get_dangerous_corridors(game_state)
        
        # If food is not in a corridor, it's safe
        if food_pos not in corridor_positions:
            return True
        
        # Food is in a corridor - check if we could escape after eating it
        walls = game_state.get_walls()
        visited = {food_pos}
        current = food_pos
        
        for _ in range(10):  # Max corridor length
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < walls.width and 0 <= ny < walls.height:
                    if not walls[nx][ny] and (nx, ny) not in visited:
                        neighbors.append((nx, ny))
            
            if not neighbors:
                break
            
            # Count exits of current position
            exits = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                       if 0 <= current[0]+dx < walls.width and 0 <= current[1]+dy < walls.height
                       and not walls[current[0]+dx][current[1]+dy])
            
            if exits > 2:  # Found junction (corridor exit)
                dist_agent_to_food = agent.get_maze_distance(agent_pos, food_pos)
                dist_food_to_exit = agent.get_maze_distance(food_pos, current)
                dist_ghost_to_exit = agent.get_maze_distance(ghost_pos, current)
                
                agent_total_dist = dist_agent_to_food + dist_food_to_exit
                if dist_ghost_to_exit <= agent_total_dist + 2:
                    return False  # Ghost could cut us off
                return True
            
            visited.add(neighbors[0])
            current = neighbors[0]
        
        # Couldn't find exit, assume unsafe if ghost is close
        return agent.get_maze_distance(agent_pos, ghost_pos) > 5
    
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
    
    def _goal_scare_retreat(self, game_state, agent, situation):
        """Scare retreat strategy: Move towards enemy base to avoid phantoms while scared."""
        walls = game_state.get_walls()
        
        # Move towards enemy base (opposite of home)
        if game_state.is_on_red_team(agent.index):
            # Red team's enemy base is right side (x=walls.width-1 area)
            target_x = walls.width - 1
        else:
            # Blue team's enemy base is left side (x=0 area)
            target_x = 0
        
        valid_targets = []
        for y in range(walls.height):
            if not walls[target_x][y]:
                valid_targets.append((target_x, y))
        
        if valid_targets:
            return min(valid_targets, 
                      key=lambda pos: agent.get_maze_distance(situation.agent_pos, pos))
        
        # Fallback to attack if no valid target
        return self._goal_attack(game_state, agent, situation)


###########################
# Layer 4: Pathfinding (A*)
###########################

class PathFinder:
    """
    A* pathfinding from current position to goal.
    Avoids walls and dangerous ghosts.
    When ghost is nearby, also avoids dead-end corridors.
    Heuristic currently is Manhattan distance to goal.
    """
    
    _corridor_cache = {}  # Class-level cache for corridor analysis
    
    @classmethod
    def get_dangerous_corridors(cls, game_state):
        """Get all positions in dead-end corridors (cached per maze layout)."""
        walls = game_state.get_walls()
        cache_key = str(walls)
        
        if cache_key in cls._corridor_cache:
            return cls._corridor_cache[cache_key]
        
        # Find dead-ends (positions with only 1 exit)
        dead_ends = set()
        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    continue
                exits = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                           if 0 <= x+dx < walls.width and 0 <= y+dy < walls.height
                           and not walls[x+dx][y+dy])
                if exits == 1:
                    dead_ends.add((x, y))
        
        # Expand dead-ends to include their corridors
        all_dangerous = set()
        for dead_end in dead_ends:
            all_dangerous.add(dead_end)
            current, visited = dead_end, {dead_end}
            
            for _ in range(5):  # Max corridor depth
                neighbors = [(current[0]+dx, current[1]+dy) 
                            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                            if 0 <= current[0]+dx < walls.width and 0 <= current[1]+dy < walls.height
                            and not walls[current[0]+dx][current[1]+dy] and (current[0]+dx, current[1]+dy) not in visited]
                
                if len(neighbors) != 1:
                    break
                
                next_pos = neighbors[0]
                exits = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                           if 0 <= next_pos[0]+dx < walls.width and 0 <= next_pos[1]+dy < walls.height
                           and not walls[next_pos[0]+dx][next_pos[1]+dy])
                
                if exits > 2:
                    break
                
                all_dangerous.add(next_pos)
                visited.add(next_pos)
                current = next_pos
        
        cls._corridor_cache[cache_key] = all_dangerous
        return all_dangerous
    
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
        
        # Identify dangerous positions and ghost proximity info
        dangers, ghost_nearby, closest_ghost_pos = self._get_danger_info(game_state, agent)
        
        # If ghost is nearby, get dangerous corridor positions to penalize
        corridor_dangers = set()
        if ghost_nearby and closest_ghost_pos:
            corridor_dangers = PathFinder.get_dangerous_corridors(game_state)
        
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
                
                # Skip if dangerous position (ghost is there)
                if next_pos in dangers:
                    continue
                
                # Calculate f = g + h (cost + heuristic)
                g = len(path) + 1
                h = agent.get_maze_distance(next_pos, goal)
                
                # Add heavy penalty for entering corridors when ghost is nearby
                corridor_penalty = 0
                if ghost_nearby and next_pos in corridor_dangers:
                    # Check if ghost could trap us in this corridor
                    ghost_dist = agent.get_maze_distance(closest_ghost_pos, next_pos)
                    our_dist = len(path) + 1
                    
                    # If ghost is closer or same distance to this corridor position, heavy penalty
                    if ghost_dist <= our_dist + 3:
                        corridor_penalty = 100  # Very high penalty to avoid this path
                
                f = g + h + corridor_penalty
                
                new_path = path + [action]
                pq.push((next_pos, new_path), f)
                
        return None
    
    def _get_danger_info(self, game_state, agent):
        """
        Get dangerous positions and ghost proximity information.
        
        Returns:
            tuple: (set of danger positions, bool if ghost is nearby, closest ghost position)
        """
        dangers = set()
        enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
        agent_pos = game_state.get_agent_position(agent.index)
        
        closest_ghost_dist = float('inf')
        closest_ghost_pos = None
        ghost_nearby = False
        
        for enemy in enemies:
            # Only dangerous if: ghost (not pacman), visible, not scared
            if not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer <= 0:
                ghost_pos = enemy.get_position()
                dist = agent.get_maze_distance(agent_pos, ghost_pos)
                
                # Track closest ghost
                if dist < closest_ghost_dist:
                    closest_ghost_dist = dist
                    closest_ghost_pos = ghost_pos
                
                # Mark ghost position as danger if close
                if dist <= 2:
                    dangers.add(ghost_pos)
                
                # Consider ghost "nearby" if within 6 steps
                if dist <= 6:
                    ghost_nearby = True
        
        return dangers, ghost_nearby, closest_ghost_pos
