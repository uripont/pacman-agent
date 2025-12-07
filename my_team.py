

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
from util import manhattan_distance

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
    """LaPulga v0.5.3, by Jorge and Oriol
    
    NEW FIX: Now correctly classifies all cells on attacking side into safe paths and cul-de-sacs.
    To be used to never again commit mistakes entering dead-end corridors.
    
    NEW PHASE: Initial static layout analysis with detailed debug output (not used for now on strategy)
    To be expanded and used during situation/strategy layers in upcoming versions.
    This takes advantage of the 15-second initialization time allowed, that we were previously ignoring.
    We expect this will allow us to better handle complex maps and remove "patches" we have been adding.

    CORE ARCHITECTURE:
    1. Situation: Analyzes game state (visible/inferred threats, food counts, team positions).
    2. Strategy: Selects high-level mode ('attack', 'defend', 'return', 'scare_retreat').
       - Now includes 'Recall' logic: attackers return home if a threat is detected.
    3. Goal: Chooses specific target coordinates based on role (Hammer vs Anvil).
    4. Pathfinding: A* with dynamic danger avoidance (avoids dead-ends if ghosts are near).
    5. Fallback: Stuck detection with strategic sacrifice to break stalemates.
    209/125/196 (39.43%) para 10 games con HARDs, antes 37%
    BENCHMARK:| 3068/4620 (66.41%)
    """
    
    DEBUG = True 
    
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
        
        # Static layout analysis (computed once, shared across team)
        self.layout_analyzer = LayoutAnalyzer()
        self.layout_info = None  # Populated in register_initial_state


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        # Perform static layout analysis (only on first agent call to avoid duplication)
        # We use a class variable to share across both agents
        # This runs within the 15-second startup allowance
        if not hasattr(LaPulga, '_layout_info_shared'):
            self.layout_info = self.layout_analyzer.analyze(game_state, self, game_state.is_on_red_team(self.index))
            LaPulga._layout_info_shared = self.layout_info  # Store in class for teammate access
            if self.DEBUG:
                self._print_layout_analysis(game_state)
        else:
            # Teammate already computed it, reuse
            self.layout_info = LaPulga._layout_info_shared
        
        # NUEVO: Inicializar memoria de comida defensiva
        self.last_defending_food = self.get_food_you_are_defending(game_state).as_list()
        self.memory_invader_pos = None # Última posición conocida/inferida del invasor
        self.memory_timer = 0          # Cuánto tiempo recordar esa posición
    
        
    def _print_layout_analysis(self, game_state):
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        
        print(f"LAYOUT ANALYSIS (Agent {self.index}) - {game_state.is_on_red_team(self.index) and 'RED' or 'BLUE'} TEAM")
        
        print(f"\nEntry groups ({len(self.layout_info.entry_groups)}):")
        for eg in self.layout_info.entry_groups:
            cells_str = ", ".join(str(c) for c in eg.cells)
            print(f"  Group {eg.id}: {eg} | Cells: {cells_str}")
        
        # NUEVO: Análisis estratégico de cápsulas
        print(f"\nCapsules Analysis (Opportunities):")
        print(f"  Own Capsules (Defending): {self.layout_info.own_capsules}")
        
        if self.layout_info.enemy_capsules:
            print(f"  Enemy Capsules Strategy:")
            # Ordenar cápsulas por valor estratégico descendente
            sorted_capsules = sorted(self.layout_info.enemy_capsules, 
                                   key=lambda c: self.layout_info.capsule_analysis[c].strategic_value, 
                                   reverse=True)
            
            for cap_pos in sorted_capsules:
                analysis = self.layout_info.capsule_analysis[cap_pos]
                cluster_info = f"Cluster {analysis.nearest_cluster.id}" if analysis.nearest_cluster else "None"
                print(f"    @ {cap_pos}: Score {analysis.strategic_value:.1f} | "
                      f"Local Food: {analysis.local_food_count} | "
                      f"Next Target -> {cluster_info} (dist: {analysis.dist_to_cluster})")
        else:
            print(f"  No Enemy Capsules detected.")

        print(f"\nOwn food clusters ({len(self.layout_info.own_clusters)}):")
        for fc in self.layout_info.own_clusters:
            food_str = ", ".join(str(f) for f in sorted(fc.initial_food))
            print(f"  {fc} | Food: {food_str}")
        
        print(f"\nEnemy food clusters ({len(self.layout_info.enemy_clusters)}):")
        for fc in self.layout_info.enemy_clusters:
            food_str = ", ".join(str(f) for f in sorted(fc.initial_food))
            # Mostramos el número de entradas y algunas de ellas
            entry_preview = list(fc.entry_cells)[:3] if fc.entry_cells else []
            entry_str = f"{entry_preview}..." if len(fc.entry_cells) > 3 else str(list(fc.entry_cells))
            print(f"  {fc} | Entries ({len(fc.entry_cells)}): {entry_str} | Food: {food_str}")

        # NUEVO: Imprimir distancias entre clusters
        print(f"\nInter-Cluster Distances (Enemy Side Connectivity):")
        if self.layout_info.cluster_distances:
            # Ordenar por ID para que sea legible
            keys = sorted(self.layout_info.cluster_distances.keys())
            for (id1, id2) in keys:
                if id1 < id2:  # Solo imprimir una vez cada par (evitar duplicados)
                    dist = self.layout_info.cluster_distances[(id1, id2)]
                    print(f"  Cluster {id1} <-> Cluster {id2}: {dist} steps")
        else:
            print("  No multi-cluster connections calculated.")
        
        # Corridor analysis results
        print(f"\nCorridor Analysis:")
        print(f"  Safe Paths (Green): {len(self.layout_info.corridor_cells)} cells")
        print(f"  Cul-de-sacs (Red): {len(self.layout_info.cul_de_sac_cells)} cells")
        print(f"\nCul-de-sac Clusters ({len(self.layout_info.cul_de_sac_clusters)}):")
        for cul_de_sac in self.layout_info.cul_de_sac_clusters:
            cells_str = ", ".join(str(c) for c in sorted(cul_de_sac.cells))
            print(f"  {cul_de_sac} | Cells: {cells_str}")
        
        # ASCII map visualization
        BLUE = "\033[44m"   # Blue background
        GREEN = "\033[42m"  # Green background
        RED = "\033[41m"    # Red background
        YELLOW = "\033[43m" # Yellow background (NEW for Capsules)
        RESET = "\033[0m"
        print(f"\nMap Visualization: {BLUE} {RESET} = Entry, {GREEN} {RESET} = Safe, {RED} {RESET} = Trap, {YELLOW}o{RESET} = Capsule")
        
        # Get all food positions
        entry_set = set(self.layout_info.boundary_cells)
        own_food = set()
        enemy_food = set()
        for cluster in self.layout_info.own_clusters:
            own_food.update(cluster.initial_food)
        for cluster in self.layout_info.enemy_clusters:
            enemy_food.update(cluster.initial_food)
        
        # Sets for capsules
        own_capsules = set(self.layout_info.own_capsules)
        enemy_capsules = set(self.layout_info.enemy_capsules)

        # Print map
        for y in range(height - 1, -1, -1):  # Print from top to bottom
            row = ""
            for x in range(width):
                if walls[x][y]:
                    row += "█"  # Wall
                elif (x, y) in entry_set:
                    row += f"{BLUE} {RESET}"  # Entry point (blue)
                
                # Check for capsules first (high priority visualization)
                elif (x, y) in own_capsules or (x, y) in enemy_capsules:
                    row += f"{YELLOW}o{RESET}"

                elif (x, y) in self.layout_info.corridor_cells:
                    # Safe path
                    if (x, y) in own_food:
                        row += f"{GREEN}●{RESET}"
                    elif (x, y) in enemy_food:
                        row += f"{GREEN}○{RESET}"
                    else:
                        row += f"{GREEN} {RESET}"
                elif (x, y) in self.layout_info.cul_de_sac_cells:
                    # Cul-de-sac
                    if (x, y) in own_food:
                        row += f"{RED}●{RESET}"
                    elif (x, y) in enemy_food:
                        row += f"{RED}○{RESET}"
                    else:
                        row += f"{RED} {RESET}"
                elif (x, y) in own_food:
                    row += "●"
                elif (x, y) in enemy_food:
                    row += "○"
                else:
                    row += " "  # Floor
            # Add y-coordinate label
            print(f"  {y:2d} | {row}")
        
        # X-axis labels (second digit - tens place)
        x_labels_tens = "     | "
        for x in range(width):
            digit = (x // 10) % 10
            x_labels_tens += str(digit) if digit != 0 else " "
        print(x_labels_tens)
        
        # X-axis labels (first digit)
        x_labels = "     | "
        for x in range(width):
            x_labels += str(x % 10)
        print(x_labels)
        
        print(f"{'='*80}\n")
            
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
        
        
    def _update_memory(self, game_state):
        """Detecta comida desaparecida y actualiza la posición inferida del enemigo"""
        current_defending_food = self.get_food_you_are_defending(game_state).as_list()
        
        # Si tenemos menos comida que antes, alguien se la ha comido
        if len(current_defending_food) < len(self.last_defending_food):
            # Encontrar qué punto de comida falta (diferencia de conjuntos)
            eaten_food = set(self.last_defending_food) - set(current_defending_food)
            
            if eaten_food:
                # El enemigo está (o estaba hace poco) en esa posición
                # Convertimos el set a lista y cogemos el primer elemento
                self.memory_invader_pos = list(eaten_food)[0]
                self.memory_timer = 20 # Recordar esto durante 20 turnos
        
        # Actualizar la lista para el siguiente turno
        self.last_defending_food = current_defending_food
        
        # Decaer el temporizador de memoria
        if self.memory_timer > 0:
            self.memory_timer -= 1
        else:
            self.memory_invader_pos = None

    # The high-level decision algorithm
    def choose_action(self, game_state):
        # NUEVO: Lógica de actualización de memoria (Antes de analizar la situación)
        self._update_memory(game_state)

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
        
        # Teammate position and vertical split
        teammates = [game_state.get_agent_state(i) for i in agent.get_team(game_state) if i != agent.index]
        teammate_pos = None
        if teammates and teammates[0].get_position() is not None:
            teammate_pos = teammates[0].get_position()
            # Determine which agent should focus on upper vs lower based on y-coord
            walls = game_state.get_walls()
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
        
        # Dangerous ghosts (non-pacman, not scared, visible)
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
        
        # Our scared state
        agent_state = game_state.get_agent_state(agent.index)
        situation.we_are_scared = agent_state.scared_timer > 0
        
        # Winning state
        score = game_state.get_score()
        if game_state.is_on_red_team(agent.index):
            situation.winning = score > 0
        else:
            situation.winning = score < 0
            
        # ---------------------------------------------------------
        # NEW DYNAMIC DEFENSE LOGIC
        # ---------------------------------------------------------
        team_indices = sorted(agent.get_team(game_state))
        
        # NUEVO: Pasar la memoria del agente a la situación
        situation.inferred_invader_pos = agent.memory_invader_pos
        situation.has_inferred_invader = (agent.memory_invader_pos is not None)
        
        # Definir si hay amenaza en el objeto situation
        situation.has_threat = situation.has_invaders_visible or situation.has_inferred_invader

        # --- LÓGICA DE ROL DEFENSIVO AVANZADA (MARTILLO Y YUNQUE) ---
        situation.is_primary_defender = True # Por defecto asumimos que somos el principal
        
        if situation.has_threat:
            target = situation.visible_invaders[0] if situation.visible_invaders else situation.inferred_invader_pos
            if target and teammate_pos:
                my_dist = agent.get_maze_distance(agent_pos, target)
                mate_dist = agent.get_maze_distance(teammate_pos, target)
                
                # Quien esté más cerca es el Defensor Primario (Martillo)
                if my_dist < mate_dist:
                    situation.is_primary_defender = True
                elif my_dist > mate_dist:
                    situation.is_primary_defender = False
                else:
                    # Desempate por índice para evitar que ambos crean ser el secundario
                    situation.is_primary_defender = (agent.index < team_indices[1]) 
        else:
             # Si no hay amenaza, definimos roles por defecto
             situation.is_primary_defender = (agent.index == team_indices[0])

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
        
        # NUEVO: Invasor inferido (no visible pero sabemos que está ahí)
        self.inferred_invader_pos = None
        self.has_inferred_invader = False
        
        self.carrying_food = 0
        self.time_remaining = 0
        
        self.scared_ghosts = []
        self.has_scared_ghosts = False
        
        self.dangerous_ghosts = []
        self.has_dangerous_ghosts = False
        self.closest_dangerous_ghost_distance = float('inf')
        self.closest_dangerous_ghost_pos = None
        
        self.we_are_scared = False
        
        self.winning = False
        self.is_defender = False


###########################
# Layer 2: Strategy
###########################

class StrategySelector:
    def select_strategy(self, situation):
        # 1. Supervivencia (Prioridad absoluta)
        if situation.we_are_scared and situation.has_invaders_visible and situation.on_own_side:
            return 'scare_retreat'
            
        # 2. Modo Depredador (Comer fantasmas asustados)
        if situation.has_scared_ghosts:
            return 'attack'
            
        # 3. Retorno Seguro (Evitar morir cargado de comida)
        # Bajamos un poco el threshold para asegurar puntos
        if situation.carrying_food >= 5: 
            return 'return'
        if situation.carrying_food > 0 and (situation.time_remaining < 60 or situation.food_remaining <= 2):
            return 'return'
            
        # 4. DEFENSA REFORZADA (CORRECCIÓN CRÍTICA)
        # Si hay amenaza, verificamos si nos toca defender AUNQUE estemos fuera
        if situation.has_threat:
            if situation.on_own_side: return 'defend'
            # Si estoy fuera pero soy el primario (el más cercano a la amenaza), vuelvo
            if situation.is_primary_defender: return 'defend'
            
        if situation.winning and situation.is_primary_defender:
            return 'defend'
        
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
        """Attack strategy: Target capsules if useful, then food, avoiding ghosts."""        
        
        # ---------------------------------------------------------
        # 1. CAPSULE LOGIC (New Priority - FIXED API CALL)
        # ---------------------------------------------------------
        # Fixed: Using get_capsules (snake_case) instead of getCapsules
        capsules = agent.get_capsules(game_state) 
        
        if capsules:
            # Find closest capsule
            closest_capsule = min(capsules, key=lambda c: agent.get_maze_distance(situation.agent_pos, c))
            dist_to_capsule = agent.get_maze_distance(situation.agent_pos, closest_capsule)
            
            # Condition A: Ghost is nearby (Panic/Weapon Mode)
            if situation.has_dangerous_ghosts and situation.closest_dangerous_ghost_distance <= 6:
                # Check if we can reach the capsule safely
                if self._is_pos_safe(game_state, agent, closest_capsule, 
                                     situation.agent_pos, situation.closest_dangerous_ghost_pos):
                    return closest_capsule
            
            # Condition B: Opportunity (It's close)
            if dist_to_capsule < 5:
                ghost_pos = situation.closest_dangerous_ghost_pos if situation.has_dangerous_ghosts else None
                if self._is_pos_safe(game_state, agent, closest_capsule, situation.agent_pos, ghost_pos):
                    return closest_capsule

        # ---------------------------------------------------------
        # 2. FOOD LOGIC (Standard)
        # ---------------------------------------------------------
        food_list = agent.get_food(game_state).as_list()
        
        # If we have teammate position info, divide territory to avoid conflicts
        if situation.should_focus_upper is not None and len(food_list) > 1:
            walls = game_state.get_walls()
            midline_y = walls.height // 2
            
            if situation.should_focus_upper:
                assigned_food = [f for f in food_list if f[1] >= midline_y]
            else:
                assigned_food = [f for f in food_list if f[1] < midline_y]
            
            if assigned_food:
                food_list = assigned_food
        
        # SAFETY CHECK: When a dangerous ghost is nearby, filter out unsafe food
        ghost_nearby_threshold = 6
        
        if situation.has_dangerous_ghosts and situation.closest_dangerous_ghost_distance <= ghost_nearby_threshold:
            ghost_pos = situation.closest_dangerous_ghost_pos
            
            # Filter food to only safe options
            safe_food = [f for f in food_list 
                        if self._is_pos_safe(game_state, agent, f, 
                                              situation.agent_pos, ghost_pos)
                        and not self._is_in_corridor_trap(game_state, agent, f,
                                                          situation.agent_pos, ghost_pos)]
            
            if safe_food:
                return min(safe_food, key=lambda f: agent.get_maze_distance(situation.agent_pos, f))
            else:
                # No safe food, return to home boundary for safety
                return self._goal_return(game_state, agent, situation)
        
        # No immediate danger, go for closest food
        if food_list:
            return min(food_list, key=lambda f: agent.get_maze_distance(situation.agent_pos, f))
        
        # Fallback if no food found
        return agent.start
    
    def _is_pos_safe(self, game_state, agent, target_pos, agent_pos, ghost_pos):
        """Check if pursuing a target position (food/capsule) is safe given a nearby ghost."""
        if ghost_pos is None:
            return True
            
        corridor_positions = PathFinder.get_dangerous_corridors(game_state)
        
        # If target is not in a corridor, it's generally safe (open field)
        if target_pos not in corridor_positions:
            return True
        
        # Target is in a corridor: compare distances to determine if we can escape
        dist_agent_to_target = agent.get_maze_distance(agent_pos, target_pos)
        dist_ghost_to_target = agent.get_maze_distance(ghost_pos, target_pos)
        
        # Safe if we can reach the target and turn around before the ghost blocks us
        return dist_ghost_to_target > dist_agent_to_target + 2
    
    def _is_in_corridor_trap(self, game_state, agent, food_pos, agent_pos, ghost_pos):
        """
        Detect if pursuing food_pos would trap us with no safe exit.
        
        Core principle: Never enter a zone where the only exit passes through ghost_pos.
        This prevents getting locked in corridors, dead-ends, or islands.
        
        Returns True if the food should be avoided (it's a trap).
        """
        if ghost_pos is None:
            return False
        
        walls = game_state.get_walls()
        
        # Count how many exits lead away from the ghost
        # We need to check if there's a path to food that doesn't require passing through ghost
        safe_exits = self._count_safe_exits(game_state, agent, food_pos, ghost_pos)
        
        # If there are NO alternative routes (all paths pass through/near ghost), it's a trap
        if safe_exits == 0:
            return True
        
        return False
    
    def _count_safe_exits(self, game_state, agent, target_pos, ghost_pos):
        """
        Count how many fundamentally different escape routes exist from target_pos
        that don't require the ghost blocking them.
        
        Uses BFS to find exits from target that lead to open areas, excluding
        paths that would be trapped if ghost moves there.
        """
        walls = game_state.get_walls()
        ghost_dist = agent.get_maze_distance(target_pos, ghost_pos)
        
        # BFS from target position to find open areas
        visited = set()
        queue = [(target_pos, 0)]  # (position, distance_from_target)
        safe_exits = 0
        MAX_SEARCH = 15  # How far to search for exits
        
        while queue:
            curr_pos, dist = queue.pop(0)
            if curr_pos in visited or dist > MAX_SEARCH:
                continue
            visited.add(curr_pos)
            
            # Count this as a safe exit if:
            # 1. It's far enough from target that ghost can't easily block both
            # 2. It's not too close to where ghost currently is
            if dist >= 5:  # Need distance to be meaningful
                curr_ghost_dist = agent.get_maze_distance(curr_pos, ghost_pos)
                # If this position is far from ghost, it's a safe exit
                if curr_ghost_dist > 8:
                    safe_exits += 1
                    if safe_exits >= 2:  # Found enough exits
                        return safe_exits
            
            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr_pos[0] + dx, curr_pos[1] + dy
                if (0 <= nx < walls.width and 0 <= ny < walls.height and 
                    not walls[nx][ny] and (nx, ny) not in visited):
                    queue.append(((nx, ny), dist + 1))
        
        return safe_exits

    def _goal_defend(self, game_state, agent, situation):
        # Objetivo prioritario: Enemigo (Visible > Inferido)
        target = None
        if situation.visible_invaders:
            target = min(situation.visible_invaders, key=lambda i: agent.get_maze_distance(situation.agent_pos, i))
        elif situation.inferred_invader_pos:
            target = situation.inferred_invader_pos
            
        if target:
            # TÁCTICA DE PINZA:
            # Si soy el Primario -> Voy directo al objetivo (Persecución)
            # Si soy el Secundario -> Voy a bloquear su salida (Intercepción)
            
            # Excepción: Si estoy muy cerca (<5 pasos), voy a matar siempre (doble ataque)
            if situation.is_primary_defender or agent.get_maze_distance(situation.agent_pos, target) < 5:
                return target
            else:
                # Lógica de "El Yunque": Ir al punto de frontera más cercano al enemigo
                walls = game_state.get_walls()
                mid_x = walls.width // 2
                boundary_x = mid_x - 1 if game_state.is_on_red_team(agent.index) else mid_x
                
                valid_boundary_spots = [(boundary_x, y) for y in range(walls.height) if not walls[boundary_x][y]]
                if valid_boundary_spots:
                    # Elegir el punto de frontera que esté más cerca del ENEMIGO, no de mí.
                    intercept_point = min(valid_boundary_spots, key=lambda p: agent.get_maze_distance(target, p))
                    return intercept_point
                return target # Fallback

        if not situation.on_own_side:
            return self._goal_return(game_state, agent, situation)
            
        # 3. Aggressive Patrol: Guard the "Frontier" Food
        defending_food = agent.get_food_you_are_defending(game_state).as_list()
        if defending_food:
            walls = game_state.get_walls()
            mid_y = walls.height // 2
            
            if game_state.is_on_red_team(agent.index):
                target_food = max(defending_food, key=lambda f: (f[0], -abs(f[1] - mid_y)))
            else:
                target_food = min(defending_food, key=lambda f: (f[0], abs(f[1] - mid_y)))
                
            return target_food
            
        return agent.start

    def _goal_return(self, game_state, agent, situation):
        """Return strategy: Go to closest home entry point."""
        walls = game_state.get_walls()
        if game_state.is_on_red_team(agent.index):
            boundary_x = 0
        else:
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
        return self._goal_attack(game_state, agent, situation)

    def _goal_scare_retreat(self, game_state, agent, situation):
        """Scare retreat strategy: Run away from enemy invaders when we're scared."""
        if not situation.visible_invaders:
            return agent.start
        
        walls = game_state.get_walls()
        valid_positions = []
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    pos = (x, y)
                    midline_x = walls.width // 2
                    if game_state.is_on_red_team(agent.index):
                        on_our_side = x < midline_x
                    else:
                        on_our_side = x >= midline_x
                    
                    if on_our_side:
                        valid_positions.append(pos)
        
        def score_position(pos):
            min_dist_to_invader = min(
                agent.get_maze_distance(pos, inv) 
                for inv in situation.visible_invaders
            )
            dist_from_current = agent.get_maze_distance(situation.agent_pos, pos)
            return min_dist_to_invader - (dist_from_current * 0.1)
        
        if valid_positions:
            nearby_positions = [
                pos for pos in valid_positions 
                if agent.get_maze_distance(situation.agent_pos, pos) <= 10
            ]
            if nearby_positions:
                best_position = max(nearby_positions, key=score_position)
                return best_position
        
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
        # ALWAYS analyze corridors to avoid walking into traps proactively
        # Don't wait until ghost is nearby - that's too late for long corridors like AlleyCapture
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
                
                # Penalize dead-end corridor entry unless it's our goal
                # This prevents entering S-shaped corridors proactively
                if next_pos in corridor_dangers and next_pos != goal:
                    # Only penalize if we're committing deeper into the corridor
                    # (i.e., this move increases our commitment to the dead-end)
                    if curr_pos not in corridor_dangers:  # Entering a corridor
                        penalty = 50  # Moderate penalty to discourage but allow if best path
                
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
        
        we_are_scared = agent_state.scared_timer > 0
        
        for enemy in enemies:
            enemy_pos = enemy.get_position()
            if enemy_pos is None: continue
            
            is_dangerous = False
            
            # CASO 1: SOMOS PACMAN (Atacando)
            if agent_state.is_pacman:
                # Nos mata cualquier fantasma no asustado
                if not enemy.is_pacman and enemy.scared_timer <= 0:
                    is_dangerous = True
            
            # CASO 2: SOMOS FANTASMA (Defendiendo)
            else:
                # Solo nos mata un Pacman SI nosotros estamos asustados
                if we_are_scared and enemy.is_pacman:
                    is_dangerous = True
                # IMPORTANTE: Si NO estamos asustados, el Pacman enemigo NO es peligroso.
                # Es comida. Al no marcarlo como danger, el A* permitirá chocar con él.

            if not is_dangerous:
                continue
            
            dist = agent.get_maze_distance(agent_pos, enemy_pos)
            if dist < closest_ghost_dist:
                closest_ghost_dist = dist
                closest_ghost_pos = enemy_pos
            
            if dist <= 2: dangers.add(enemy_pos)
            if dist <= 6: ghost_nearby = True
        
        return dangers, ghost_nearby, closest_ghost_pos


###########################
# Layout Static Analysis  #
###########################

class EntryGroup:
    """Represents a contiguous group of entry points (a 'line' of attack)"""
    def __init__(self, group_id, cells):
        self.id = group_id
        self.cells = cells  # List of (x,y) boundary cells
        self.representative = cells[len(cells) // 2]  # Middle cell
        y_coords = [c[1] for c in cells]
        self.y_range = (min(y_coords), max(y_coords))
    
    def __repr__(self):
        return f"EntryGroup(id={self.id}, cells={len(self.cells)}, rep={self.representative})"


class FoodCluster:
    """Represents a spatial cluster of food pellets"""
    def __init__(self, cluster_id, initial_food_set, side):
        self.id = cluster_id
        self.initial_food = initial_food_set  # Set of (x,y)
        self.side = side    # 'own' or 'enemy'
        self.entry_cells = set()  # NUEVO: Celdas vacías desde las que se accede al cluster
    
    def count_remaining(self, game_state, agent):
        """Count how many food pellets are still present"""
        if self.side == 'own':
            current_food = agent.get_food_you_are_defending(game_state).as_list()
        else:
            current_food = agent.get_food(game_state).as_list()
        return len(self.initial_food & set(current_food))
    
    def __repr__(self):
        return f"FoodCluster(id={self.id}, size={len(self.initial_food)}, entries={len(self.entry_cells)})"


class CulDeSac: # TODO: should replace all corridor analysis we were doing before
    """Represents a dead-end zone with its entry point"""
    def __init__(self, sac_id, cells, entry_cell, depth):
        self.id = sac_id
        self.cells = cells  # Set of (x,y) in this cul-de-sac
        self.entry_cell = entry_cell  # The chokepoint (x,y)
        self.depth = depth  # Max distance from entry to deepest cell
    
    def __repr__(self):
        return f"CulDeSac(id={self.id}, size={len(self.cells)}, entry={self.entry_cell}, depth={self.depth})"


class CellInfo:
    """Per-cell metadata about escape routes and safety"""
    def __init__(self, pos):
        self.pos = pos
        self.is_cul_de_sac = False
        self.cul_de_sac_id = None
        self.distance_to_home = float('inf')
        self.nearest_home_entry = None
        self.escape_cells = set()  # Cells that lead toward home
        self.num_exits = 0  # How many distinct escape directions
        self.is_chokepoint = False  # Is this the entry to a cul-de-sac?
    
    def __repr__(self):
        return f"CellInfo({self.pos}, cul={self.is_cul_de_sac}, exits={self.num_exits})"


class CapsuleAnalysis:
    """Stores strategic value of a specific power capsule"""
    def __init__(self, pos):
        self.pos = pos
        self.local_food_count = 0      # Food within immediate reach (e.g., 10 steps)
        self.nearest_cluster = None    # The best FoodCluster to attack after this
        self.dist_to_cluster = float('inf')
        self.strategic_value = 0.0     # Heuristic score
    
    def __repr__(self):
        cluster_id = self.nearest_cluster.id if self.nearest_cluster else "None"
        return f"Capsule({self.pos}): Value={self.strategic_value:.1f}, LocalFood={self.local_food_count}, Aim->Cluster {cluster_id}"


class LayoutInfo:
    """Container for all static layout analysis"""
    def __init__(self):
        # Home entry analysis
        self.boundary_cells = []  # All passable cells on the boundary
        self.entry_groups = []  # List of EntryGroup
        self.cell_to_entry_group = {}  # Dict: (x,y) to EntryGroup
        
        # Food cluster analysis
        self.own_clusters = []  # List of FoodCluster on our side
        self.enemy_clusters = []  # List of FoodCluster on enemy side
        self.cell_to_cluster = {}  # Dict: (x,y) to FoodCluster
        
        # NUEVO: Matriz de distancias entre clusters
        # Keys: tupla (id_cluster_A, id_cluster_B) -> Value: int (distancia)
        self.cluster_distances = {}
        
        # Capsule analysis
        self.own_capsules = []    # List of (x,y)
        self.enemy_capsules = []  # List of (x,y)
        self.capsule_analysis = {} # Dict: (x,y) -> CapsuleAnalysis object (NUEVO)
        
        # Corridor analysis
        self.corridor_cells = set()  # Safe paths (has redundant routes home)
        self.cul_de_sac_cells = set()  # Dead-end zones
        self.cul_de_sac_clusters = []  # List of CulDeSac clusters
        self.cell_to_safety_type = {}  # Dict: (x,y) to 'safe_path' or 'cul_de_sac'

class LayoutAnalyzer:
    """Performs static analysis of the game layout (one-time at game start)"""
    
    def analyze(self, game_state, agent, team_is_red):
        """Main entry point: perform all layout analyses"""
        layout_info = LayoutInfo()
        walls = game_state.get_walls()
        
        # Phase 1: Analyze home entry points and group them
        self._analyze_home_entries(layout_info, walls, team_is_red)
        
        # Phase 2: Analyze food clusters AND their entry points (ACTUALIZADO)
        self._analyze_food_clusters(layout_info, game_state, agent, team_is_red)
        
        # Phase 3: Pre-calculate distances between enemy clusters (NUEVO)
        # Solo calculamos distancias entre clusters enemigos porque es donde nos moveremos atacando
        self._analyze_cluster_distances(layout_info, walls)

        # Phase 4: Analyze power capsules and their potential
        self._analyze_capsules(layout_info, game_state, agent)
        self._analyze_capsule_potential(layout_info, walls, agent)

        # Phase 5: Analyze corridors and cul-de-sacs on attack side
        self._analyze_corridors_and_culs_de_sac(layout_info, walls, team_is_red)
        
        return layout_info

    def _analyze_capsules(self, layout_info, game_state, agent):
        """Identify power capsules positions."""
        layout_info.own_capsules = agent.get_capsules_you_are_defending(game_state)
        layout_info.enemy_capsules = agent.get_capsules(game_state)

    def _analyze_capsule_potential(self, layout_info, walls, agent):
        """
        Calculates the strategic value of each enemy capsule.
        Determines the best next move (target cluster) after eating it.
        """
        for cap_pos in layout_info.enemy_capsules:
            analysis = CapsuleAnalysis(cap_pos)
            
            # 1. Calculate Local Density: How much food is within 10 steps?
            # This represents the "immediate reward" while invulnerable.
            nearby_food = 0
            search_depth = 10
            
            # Simple BFS for local food density
            queue = [(cap_pos, 0)]
            visited = {cap_pos}
            
            while queue:
                curr, dist = queue.pop(0)
                if dist > search_depth:
                    continue
                
                # Check if this cell has food (using our static cluster map)
                if curr in layout_info.cell_to_cluster:
                    cluster = layout_info.cell_to_cluster[curr]
                    if cluster.side == 'enemy': # Should be, but safety check
                        nearby_food += 1
                
                if dist < search_depth:
                    for neighbor in self._get_neighbors(curr, walls):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
            
            analysis.local_food_count = nearby_food
            
            # 2. Identify Best Cluster Target
            # Which large food cluster is closest to this capsule?
            best_cluster = None
            min_dist = float('inf')
            
            for cluster in layout_info.enemy_clusters:
                # Find distance from capsule to the NEAREST piece of food in this cluster
                # We use Manhattan for speed in static analysis, or actual maze dist if careful
                # Here we estimate using min maze distance to cluster members
                
                # Optimización: comprobar distancia al centroide o a un punto representativo
                # Para precisión, buscamos la comida más cercana del cluster
                dist_to_cluster = float('inf')
                for food_pos in cluster.initial_food:
                    d = abs(cap_pos[0] - food_pos[0]) + abs(cap_pos[1] - food_pos[1]) # Manhattan approx
                    if d < dist_to_cluster:
                        dist_to_cluster = d
                
                if dist_to_cluster < min_dist:
                    min_dist = dist_to_cluster
                    best_cluster = cluster
            
            analysis.nearest_cluster = best_cluster
            analysis.dist_to_cluster = min_dist
            
            # 3. Calculate Strategic Score
            # Formula: (Local Food * 2) + (Cluster Size / (Distance to Cluster + 1))
            # Rewards capsules surrounded by food or close to big clusters.
            cluster_value = 0
            if best_cluster:
                cluster_value = len(best_cluster.initial_food) / (min_dist + 1)
            
            analysis.strategic_value = (nearby_food * 2.0) + cluster_value
            
            # Store analysis
            layout_info.capsule_analysis[cap_pos] = analysis

    def _analyze_cluster_distances(self, layout_info, walls):
        """
        NUEVO: Pre-calculate distances between all pairs of Enemy Clusters.
        Stores the Maze Distance between the ENTRY CELLS of Cluster A and Cluster B.
        """
        clusters = layout_info.enemy_clusters
        n = len(clusters)
        
        # Inicializar diccionario
        layout_info.cluster_distances = {}
        
        # Comparar cada par de clusters (sin repetir)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = clusters[i]
                c2 = clusters[j]
                
                # Calcular la distancia más corta entre el CONJUNTO de entradas de c1
                # y el CONJUNTO de entradas de c2.
                dist = self._bfs_distance_between_sets(c1.entry_cells, c2.entry_cells, walls)
                
                # Guardar en ambas direcciones
                layout_info.cluster_distances[(c1.id, c2.id)] = dist
                layout_info.cluster_distances[(c2.id, c1.id)] = dist

    def _bfs_distance_between_sets(self, start_set, goal_set, walls):
        """
        Calculates shortest path from ANY cell in start_set to ANY cell in goal_set.
        Returns distance as integer.
        """
        if not start_set or not goal_set:
            return float('inf')
            
        # Optimization: Check if sets overlap (distance 0)
        if not start_set.isdisjoint(goal_set):
            return 0
            
        # BFS Init
        queue = []
        visited = set()
        
        # Añadir todas las celdas de inicio con distancia 0
        for cell in start_set:
            queue.append((cell, 0))
            visited.add(cell)
            
        while queue:
            curr_pos, dist = queue.pop(0)
            
            # Si llegamos a CUALQUIER celda del objetivo, hemos terminado
            if curr_pos in goal_set:
                return dist
                
            # Expandir
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr_pos[0] + dx, curr_pos[1] + dy
                
                if (0 <= nx < walls.width and 0 <= ny < walls.height and 
                    not walls[nx][ny] and (nx, ny) not in visited):
                    
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        
        return float('inf')  # No conectadas
    
    def _analyze_home_entries(self, layout_info, walls, team_is_red):
        """
        Identify all entry points to home base and group contiguous ones.
        Entry groups are "lines" of adjacent entry points (important for interception).
        """
        width = walls.width
        height = walls.height
        
        # Determine boundary x coordinate
        mid_x = width // 2
        boundary_x = mid_x - 1 if team_is_red else mid_x
        
        # Find all passable boundary cells
        boundary_cells = []
        for y in range(height):
            if not walls[boundary_x][y]:
                boundary_cells.append((boundary_x, y))
        
        layout_info.boundary_cells = boundary_cells
        
        # Group contiguous boundary cells into entry groups
        visited = set()
        entry_group_id = 0
        
        for cell in boundary_cells:
            if cell in visited:
                continue
            
            # BFS to find all contiguous cells
            group_cells = []
            queue = [cell]
            visited.add(cell)
            
            while queue:
                curr = queue.pop(0)
                group_cells.append(curr)
                
                # Look for adjacent boundary cells (vertical neighbors only, since we're on a line)
                for neighbor in [(curr[0], curr[1] - 1), (curr[0], curr[1] + 1)]:
                    if neighbor in boundary_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Create entry group
            entry_group = EntryGroup(entry_group_id, group_cells)
            layout_info.entry_groups.append(entry_group)
            entry_group_id += 1
            
            # Map each cell to its group
            for cell in group_cells:
                layout_info.cell_to_entry_group[cell] = entry_group
    
    def _analyze_food_clusters(self, layout_info, game_state, agent, team_is_red):
        """
        Identifies food clusters and calculates their ENTRY CELLS.
        An entry cell is a non-wall, non-cluster cell adjacent to any food in the cluster.
        """
        walls = game_state.get_walls()
        own_side_food = set(agent.get_food_you_are_defending(game_state).as_list())
        all_food = set(agent.get_food(game_state).as_list())
        enemy_side_food = all_food
        
        layout_info.own_clusters = self._cluster_contiguous_food(own_side_food, walls)
        layout_info.enemy_clusters = self._cluster_contiguous_food(enemy_side_food, walls)
        
        # Assign sides
        for cluster in layout_info.own_clusters:
            cluster.side = 'own'
        for cluster in layout_info.enemy_clusters:
            cluster.side = 'enemy'
        
        # Global map mapping
        for cluster in layout_info.own_clusters + layout_info.enemy_clusters:
            for food_pos in cluster.initial_food:
                layout_info.cell_to_cluster[food_pos] = cluster

        # NUEVO: Calcular Entry Cells para cada cluster
        # Analizamos tanto los nuestros (para defender entradas) como los enemigos (para atacar)
        all_clusters = layout_info.own_clusters + layout_info.enemy_clusters
        
        for cluster in all_clusters:
            entry_cells = set()
            for food_pos in cluster.initial_food:
                # Mirar vecinos
                neighbors = self._get_neighbors(food_pos, walls)
                for n in neighbors:
                    # Si el vecino NO es parte de este cluster (es espacio vacío o comida de otro grupo)
                    # lo consideramos una entrada.
                    if n not in cluster.initial_food:
                        entry_cells.add(n)
            
            cluster.entry_cells = entry_cells
    
    def _cluster_contiguous_food(self, food_set, walls):
        """
        Cluster food positions using contiguous adjacency.
        Two food are in same cluster if they are directly adjacent (4-connected).
        """
        if not food_set:
            return []
        
        visited = set()
        clusters = []
        cluster_id = 0
        
        for food in food_set:
            if food in visited:
                continue
            
            # BFS to find contiguous group
            cluster_cells = set()
            queue = [food]
            visited.add(food)
            
            while queue:
                curr = queue.pop(0)
                cluster_cells.add(curr)
                
                # Find adjacent food cells
                for neighbor in self._get_neighbors(curr, walls):
                    if neighbor in food_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            cluster = FoodCluster(cluster_id, cluster_cells, 'unknown')
            clusters.append(cluster)
            cluster_id += 1
        
        return clusters
    
    def _get_neighbors(self, pos, walls):
        """Get all valid non-wall neighbors of a position"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                neighbors.append((nx, ny))
        return neighbors
    
    def _cluster_contiguous_cells(self, cells):
        """Group a set of cells into contiguous clusters (4-connected)"""
        if not cells:
            return []
        
        visited = set()
        clusters = []
        
        for cell in cells:
            if cell in visited:
                continue
            
            # BFS to find contiguous group
            cluster = set()
            queue = [cell]
            visited.add(cell)
            
            while queue:
                curr = queue.pop(0)
                cluster.add(curr)
                
                # Find adjacent cells in the same set
                for neighbor in [(curr[0]+1, curr[1]), (curr[0]-1, curr[1]), 
                                (curr[0], curr[1]+1), (curr[0], curr[1]-1)]:
                    if neighbor in cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            clusters.append(cluster)
        
        return clusters
    
    def _analyze_corridors_and_culs_de_sac(self, layout_info, walls, team_is_red):
        """
        Classify passable cells on attack side into:
        - Safe paths (green): cells with independent routes to boundary (not blocked by single chokepoint)
        - Cul-de-sacs (red): cells that all route through a single chokepoint to escape
        
        Algorithm we used:
        1. Identify dead-ends (cells with only 1 neighbor)
        2. Iteratively expand: treat identified cul-de-sacs as walls (!), find new dead-ends behind them
        3. Repeat until no more cul-de-sacs are found, and thus we have also the safe path
        """
        width = walls.width
        height = walls.height
        mid_x = width // 2
        
        # Determine which side is attack
        if team_is_red:
            attack_side_range = range(mid_x, width)
            boundary_x = mid_x
        else:
            attack_side_range = range(0, mid_x)
            boundary_x = mid_x - 1
        
        # Build the passable graph on attack side
        passable_attack = set()
        for x in attack_side_range:
            for y in range(height):
                if not walls[x][y]:
                    passable_attack.add((x, y))
        
        # Add food cells to the graph
        for cluster in layout_info.enemy_clusters:
            passable_attack.update(cluster.initial_food)
        
        # Iteratively identify cul-de-sacs by treating previous cul-de-sacs as walls
        cul_de_sac_cells = set()
        max_iterations = len(passable_attack)  # Worst case: all cells are cul-de-sacs
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            available = passable_attack - cul_de_sac_cells
            
            if not available:
                break
            
            new_dead_ends = set()
            for cell in available:
                # Count neighbors that are exits
                # An exit is either:
                # 1. Another available cell (path deeper into enemy territory)
                # 2. A safe cell (Home territory) - CRITICAL FIX: Don't treat home as a wall!
                
                neighbors = self._get_neighbors(cell, walls)
                exit_count = 0
                
                for n in neighbors:
                    if n in available:
                        exit_count += 1
                    elif n not in passable_attack:
                        # Neighbor is not in attack set, but is a valid non-wall.
                        # This implies it's on our Home side.
                        # Being connected to Home is the ultimate safety, so it counts as an exit.
                        exit_count += 1
                
                # If only 0 or 1 exit, it's a dead end (cul-de-sac)
                if exit_count <= 1:
                    new_dead_ends.add(cell)
            
            if not new_dead_ends:
                break
            
            cul_de_sac_cells.update(new_dead_ends)
        
        # Safe paths are everything else
        safe_cells = passable_attack - cul_de_sac_cells
        layout_info.corridor_cells = safe_cells
        layout_info.cul_de_sac_cells = cul_de_sac_cells
        
        # Cluster the cul-de-sac cells into contiguous zones
        layout_info.cul_de_sac_clusters = self._cluster_contiguous_cells(cul_de_sac_cells)
        # Convert to CulDeSac objects
        cul_de_sac_objs = []
        for cluster_id, cells in enumerate(layout_info.cul_de_sac_clusters):
            cul_de_sac = CulDeSac(cluster_id, cells, None, 0)  # entry_cell and depth set to None/0 for now
            cul_de_sac_objs.append(cul_de_sac)
        layout_info.cul_de_sac_clusters = cul_de_sac_objs
        
        # Build cell-to-type mapping
        for cell in layout_info.corridor_cells:
            layout_info.cell_to_safety_type[cell] = 'safe_path'
        for cell in layout_info.cul_de_sac_cells:
            layout_info.cell_to_safety_type[cell] = 'cul_de_sac'

