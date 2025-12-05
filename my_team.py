# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util as util

from capture_agents import CaptureAgent
from game import Directions

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='LaPulga', second='LaPulga', num_training=0): # We use our LaPulga agent for both slots
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
    """LaPulga v0.4.6, by Jorge and Oriol
    

    
    CURRENT APPROACH: Four-layer decision system:
    1. Situation: Detects game state (position, food, threats, scared state, etc.)
    2. Strategy: Maps certain attributes of the situation to a strategy mode
       - 'scare_retreat' when we're scared ghosts being pursued by invaders
    3. Goal: Chooses target position based on strategy
    4. Pathfinding: A* to find optimal path to goal
       - Avoids walls and dangerous enemies (ghosts when attacking, invaders when scared)
       - Heuristic is Manhattan distance
    5. Fallback: When no path exists, detects if stuck and sacrifices if losing
        
    NEW FIX: Stuck detection and strategic sacrifice 
    NEW FIX: Improved scared ghost behavior 
    """
    
    # Set to True to see debug output about scared retreat behavior and stuck detection
    DEBUG = True  # Enabled to debug stuck detection
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        
        # State vars
        self.start = None
        
        # Stuck detection (for fallback suicide mechanism)
        self.consecutive_stops = 0
        self.last_position = None
        self.position_history = []  # Track last N positions to detect oscillation
        
        # Initialize layers
        self.situation_analyzer = SituationAnalyzer()
        self.strategy_selector = StrategySelector()
        self.goal_chooser = GoalChooser()
        self.path_finder = PathFinder()

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
    def _should_sacrifice(self, game_state):
        """Determine if we should sacrifice ourselves when stuck.
        
        Only sacrifice if we're losing or tied. If winning, staying stuck is good.
        """
        score = game_state.get_score()
        
        # Adjust score based on team (red team has positive score when winning)
        if not game_state.is_on_red_team(self.index):
            score = -score
        
        # Sacrifice if we're losing or tied
        # If score > 0, we're winning, so don't sacrifice
        return score <= 0
    
    def _get_suicide_action(self, game_state):
        """Get an action that moves us towards the nearest enemy (to get killed).
        
        This breaks stalemates where we're stuck in a corridor.
        """
        current_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        # Find nearest visible enemy
        visible_enemies = [e for e in enemies if e.get_position() is not None]
        if not visible_enemies:
            # No visible enemies, just try to move in any valid direction
            return self._get_any_valid_action(game_state)
        
        # Get closest enemy position
        enemy_positions = [e.get_position() for e in visible_enemies]
        closest_enemy = min(enemy_positions, 
                          key=lambda pos: self.get_maze_distance(current_pos, pos))
        
        # Find the action that moves us closest to the enemy
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)  # Don't stay still
        
        if not legal_actions:
            return Directions.STOP
        
        # Choose action that minimizes distance to enemy
        best_action = None
        best_dist = float('inf')
        
        for action in legal_actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(new_pos, closest_enemy)
            
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action if best_action else Directions.STOP
    
    def _get_any_valid_action(self, game_state):
        """Get any valid action when we can't see enemies."""
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        
        return legal_actions[0] if legal_actions else Directions.STOP
    
    def _is_oscillating(self):
        """Detect if agent is oscillating in a small area (trapped but moving).
        
        Returns True if all positions in history are within a small radius.
        """
        if len(self.position_history) < 8:  # Reduced from 10 for faster detection
            return False
        
        # Calculate the bounding box of recent positions
        x_coords = [pos[0] for pos in self.position_history]
        y_coords = [pos[1] for pos in self.position_history]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # If moving in area smaller than 2x2, we're oscillating/trapped
        OSCILLATION_THRESHOLD = 2  # Reduced from 3 for more aggressive detection
        return x_range <= OSCILLATION_THRESHOLD and y_range <= OSCILLATION_THRESHOLD
        
        
    # The high-level decision algorithm
    def choose_action(self, game_state):
        self.situation = self.situation_analyzer.analyze(game_state, self)
        self.strategy = self.strategy_selector.select_strategy(self.situation)
        self.goal = self.goal_chooser.choose_goal(game_state, self, self.situation, self.strategy)
        path = self.path_finder.search(game_state, self, self.goal)
        
        current_pos = game_state.get_agent_position(self.index)
        
        # Update position history for oscillation detection
        self.position_history.append(current_pos)
        if len(self.position_history) > 10:  # Keep last 10 positions
            self.position_history.pop(0)
        
        if path:
            # Reset stuck counter when we have a valid path
            self.consecutive_stops = 0
            self.last_position = current_pos
            return path[0]
        
        # No path found - we're in fallback mode
        # Check if we're stuck (not moving for multiple turns)
        if current_pos == self.last_position:
            self.consecutive_stops += 1
        else:
            self.consecutive_stops = 0
        
        self.last_position = current_pos
        
        # STUCK DETECTION: Check both STOP and oscillation patterns
        STUCK_THRESHOLD = 5
        is_stuck = False
        
        # Method 1: Consecutive STOPs
        if self.consecutive_stops >= STUCK_THRESHOLD:
            is_stuck = True
            if self.DEBUG:
                print(f"Agent {self.index}: Detected STUCK (consecutive STOPs: {self.consecutive_stops})")
        
        # Method 2: Oscillating in small area (moving back and forth)
        if len(self.position_history) >= 8:  # Match the threshold in _is_oscillating
            if self._is_oscillating():
                is_stuck = True
                if self.DEBUG:
                    print(f"Agent {self.index}: Detected STUCK (oscillating in small area)")
        
        if is_stuck:
            # Check if we should sacrifice ourselves
            if self._should_sacrifice(game_state):
                # Force suicide by moving towards nearest enemy
                suicide_action = self._get_suicide_action(game_state)
                if suicide_action:
                    if self.DEBUG:
                        print(f"Agent {self.index}: SACRIFICING (losing game, score: {game_state.get_score()})")
                    return suicide_action
            else:
                # We're winning, staying stuck is good for us
                if self.DEBUG:
                    print(f"Agent {self.index}: STUCK but WINNING (score: {game_state.get_score()}) - staying put")
        
        # Fallback: return STOP
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
        food_list = agent.get_food(game_state).as_list()
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
        situation.carrying_food = game_state.get_agent_state(agent.index).num_carrying
        
        # Time remaining
        situation.time_remaining = game_state.data.timeleft
        
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
        
        # Our scared state (when enemy eats power capsule, we become scared ghosts)
        agent_state = game_state.get_agent_state(agent.index)
        situation.we_are_scared = agent_state.scared_timer > 0
        
        return situation
    
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
        self.should_focus_upper = None
        
        self.food_remaining = 0
        self.closest_food_pos = None
        
        self.visible_invaders = []
        self.has_invaders_visible = False
        
        self.carrying_food = 0
        self.time_remaining = 0
        
        self.scared_ghosts = []
        self.has_scared_ghosts = False
        
        self.dangerous_ghosts = []
        self.has_dangerous_ghosts = False
        self.closest_dangerous_ghost_distance = float('inf')
        self.closest_dangerous_ghost_pos = None
        
        self.we_are_scared = False


###########################
# Layer 2: Strategy
###########################

class StrategySelector:
    """Select strategy based on attribute combination of the current situation."""

    def select_strategy(self, situation):
        if situation.we_are_scared and situation.has_invaders_visible and situation.on_own_side:
            return 'scare_retreat'
        if situation.has_invaders_visible and situation.on_own_side:
            return 'defend'
        if situation.carrying_food >= 6: # Threshold to return with food
            return 'return'
        if situation.carrying_food > 0 and situation.time_remaining < 150: # Come back when time is low
            return 'return'
        if situation.carrying_food > 0 and situation.food_remaining <= 2: # Ignore last 2 food
            return 'return'
        if situation.has_scared_ghosts and not situation.on_own_side:
            return 'hunt_scared'
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
        if strategy == 'attack':
            return self._goal_attack(game_state, agent, situation)
        elif strategy == 'defend':
            return self._goal_defend(game_state, agent, situation)
        elif strategy == 'return':
            return self._goal_return(game_state, agent, situation)
        elif strategy == 'hunt_scared':
            return self._goal_hunt_scared(game_state, agent, situation)
        else:  # strategy == 'scare_retreat':
            return self._goal_scare_retreat(game_state, agent, situation)
    
    def _goal_attack(self, game_state, agent, situation):
        """Attack strategy: Go for closest food, preferring safe food when ghost is nearby."""        
        food_list = agent.get_food(game_state).as_list()
        
        # If we have teammate position info, divide territory to avoid conflicts
        if situation.should_focus_upper is not None and len(food_list) > 1:
            # Split food into upper and lower halves
            walls = game_state.get_walls()
            midline_y = walls.height // 2
            
            # Partition food based on vertical position
            if situation.should_focus_upper:
                assigned_food = [f for f in food_list if f[1] >= midline_y]
            else:
                assigned_food = [f for f in food_list if f[1] < midline_y]
            
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
                return min(safe_food, key=lambda f: agent.get_maze_distance(situation.agent_pos, f))
            else:
                # No safe food, return to home boundary for safety
                return self._goal_return(game_state, agent, situation)
        
        # No immediate danger, go for closest food
        return min(food_list, key=lambda f: agent.get_maze_distance(situation.agent_pos, f))
    
    def _is_food_safe(self, game_state, agent, food_pos, agent_pos, ghost_pos):
        """Check if pursuing food is safe given a nearby ghost."""
        corridor_positions = PathFinder.get_dangerous_corridors(game_state)
        
        # If food is not in a corridor, it's safe
        if food_pos not in corridor_positions:
            return True
        
        # Food is in a corridor: compare distances to determine if we can escape
        # We need to reach the food and get out before the ghost catches us
        dist_agent_to_food = agent.get_maze_distance(agent_pos, food_pos)
        dist_ghost_to_food = agent.get_maze_distance(ghost_pos, food_pos)
        
        # Safe if we can reach the food much before the ghost
        # (margin of 2 to account for exiting the corridor)
        return dist_ghost_to_food > dist_agent_to_food + 2
    
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
            return min(valid_entries, key=lambda pos: agent.get_maze_distance(situation.agent_pos, pos))
        
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
        """Scare retreat strategy: Run away from enemy invaders when we're scared."""
        if not situation.visible_invaders:
            # No visible invaders, just stay safe near our spawn
            return agent.start
        
        # Find a safe position that maximizes distance from all invaders
        walls = game_state.get_walls()
        
        # Get all valid positions on our side
        valid_positions = []
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    pos = (x, y)
                    # Check if position is on our side
                    agent_state_at_pos = game_state.get_agent_state(agent.index)
                    # Use the midline to determine our side
                    midline_x = walls.width // 2
                    if game_state.is_on_red_team(agent.index):
                        on_our_side = x < midline_x
                    else:
                        on_our_side = x >= midline_x
                    
                    if on_our_side:
                        valid_positions.append(pos)
        
        # Score each position by minimum distance to any invader (we want maximum of this)
        def score_position(pos):
            min_dist_to_invader = min(
                agent.get_maze_distance(pos, inv) 
                for inv in situation.visible_invaders
            )
            # Also consider distance from our current position (don't want to go too far)
            dist_from_current = agent.get_maze_distance(situation.agent_pos, pos)
            # Prefer positions far from invaders but not too far from current position
            return min_dist_to_invader - (dist_from_current * 0.1)
        
        # Find the best safe position
        if valid_positions:
            # Limit search to nearby positions for performance
            nearby_positions = [
                pos for pos in valid_positions 
                if agent.get_maze_distance(situation.agent_pos, pos) <= 10
            ]
            if nearby_positions:
                best_position = max(nearby_positions, key=score_position)
                return best_position
        
        # Fallback to spawn point
        return agent.start


###########################
# Layer 4: Pathfinding (A*)
###########################

class PathFinder:
    """
    A* pathfinding from current position to goal.
    Avoids walls and dangerous ghosts.
    When ghost is nearby, also avoids dead-end corridors.
    """
    
    _corridor_cache = {}  # Cache for corridor analysis
    
    @classmethod
    def get_dangerous_corridors(cls, game_state):
        """Get all positions in dead-end/cul de sac corridors (cached per maze layout, so only once at the beginnnig)"""
        walls = game_state.get_walls()
        cache_key = str(walls)
        if cache_key in cls._corridor_cache:
            return cls._corridor_cache[cache_key]
        
        def count_exits(x, y):
            return sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                      if 0 <= x+dx < walls.width and 0 <= y+dy < walls.height
                      and not walls[x+dx][y+dy])
        
        # Find dead-ends (positions with only 1 exit)
        dead_ends = {(x, y) for x in range(walls.width) for y in range(walls.height)
                    if not walls[x][y] and count_exits(x, y) == 1}
        
        # Expand dead-ends to include  corridors
        all_dangerous = set()
        for dead_end in dead_ends:
            all_dangerous.add(dead_end)
            current, visited = dead_end, {dead_end}
            
            for _ in range(5):  # Max corridor depth
                neighbors = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = current[0] + dx, current[1] + dy
                    in_bounds = 0 <= nx < walls.width and 0 <= ny < walls.height
                    not_wall = not walls[nx][ny]
                    not_visited = (nx, ny) not in visited
                    
                    if in_bounds and not_wall and not_visited:
                        neighbors.append((nx, ny))
                
                if len(neighbors) != 1 or count_exits(neighbors[0][0], neighbors[0][1]) > 2:
                    break
                
                current = neighbors[0]
                all_dangerous.add(current)
                visited.add(current)
        
        cls._corridor_cache[cache_key] = all_dangerous
        return all_dangerous
    
    def search(self, game_state, agent, goal):
        """Find optimal path to goal using A* search avoiding ghosts and corridors."""
        start_pos = game_state.get_agent_position(agent.index)
        pq = util.PriorityQueue()
        pq.push((start_pos, []), 0)
        visited = set()
        
        # Get threat analysis to avoid ghosts
        dangers, ghost_nearby, closest_ghost_pos = self._get_danger_info(game_state, agent)
        corridor_dangers = PathFinder.get_dangerous_corridors(game_state) if ghost_nearby else set()
        
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
            
            # Explore neighbors
            for dx, dy, action in [(0, 1, Directions.NORTH), (0, -1, Directions.SOUTH),
                                   (1, 0, Directions.EAST), (-1, 0, Directions.WEST)]:
                next_x, next_y = int(curr_pos[0] + dx), int(curr_pos[1] + dy)
                next_pos = (next_x, next_y)
                
                # Skip walls and immediate ghost positions
                if game_state.get_walls()[next_x][next_y] or next_pos in dangers:
                    continue
                
                # Calculate A* score: g (cost) + h (heuristic) + corridor penalty
                g = len(path) + 1
                h = agent.get_maze_distance(next_pos, goal) 
                penalty = 0
                
                # Penalize corridor entry if ghost is nearby
                if ghost_nearby and next_pos in corridor_dangers:
                    ghost_dist = agent.get_maze_distance(closest_ghost_pos, next_pos)
                    if ghost_dist <= g + 3:  # Ghost could trap us
                        penalty = 100
                
                f = g + h + penalty
                pq.push((next_pos, path + [action]), f)
        
        return None
    
    def _get_danger_info(self, game_state, agent):
        dangers = set()
        ghost_nearby = False
        closest_ghost_pos = None
        closest_ghost_dist = float('inf')
        
        agent_pos = game_state.get_agent_position(agent.index)
        agent_state = game_state.get_agent_state(agent.index)
        enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
        
        # Check if we're scared
        we_are_scared = agent_state.scared_timer > 0
        
        for enemy in enemies:
            enemy_pos = enemy.get_position()
            if enemy_pos is None:
                continue
            
            # Determine if this enemy is dangerous to us
            is_dangerous = False
            
            if we_are_scared and enemy.is_pacman:
                # When we're scared, enemy pacmen can kill us
                is_dangerous = True
            elif not enemy.is_pacman and enemy.scared_timer <= 0:
                # When we're attacking, non-scared enemy ghosts can kill us
                is_dangerous = True
            
            if not is_dangerous:
                continue
            
            dist = agent.get_maze_distance(agent_pos, enemy_pos)
            
            # Track closest dangerous enemy
            if dist < closest_ghost_dist:
                closest_ghost_dist = dist
                closest_ghost_pos = enemy_pos
            
            # Mark as dangerous if very close
            if dist <= 2:
                dangers.add(enemy_pos)
            
            # Mark nearby if moderately close
            if dist <= 6:
                ghost_nearby = True
        
        return dangers, ghost_nearby, closest_ghost_pos
