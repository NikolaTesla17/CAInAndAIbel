#by Absalom Ranelletti
#
#
# Winter 2026
from pacai.core.action import Action
from pacai.capture.gamestate import GameState
import pacai.core.agentinfo
import pacai.capture.team
import pacai.util.alias
import pacai.core.agent
import typing
import pacai.core.gamestate
import pacai.search.distance
MAX_DEPTH = 3

# constants for the copied dummy feature extractor
CLOSE_GHOST_DISTANCE: float = 1.0
CLOSE_FOOD_DISTANCE: float = 0.0

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    # return pacai.capture.team.create_team_dummy()
    
    agent1_info = pacai.core.agentinfo.AgentInfo(f"{__name__}.Cain")
    agent2_info = pacai.core.agentinfo.AgentInfo(f"{__name__}.Abel")

    return [agent1_info, agent2_info]

#skeleton parent to Cain and Abel
class Eve(pacai.core.agent.Agent):
    def __init__(self,
            **kwargs: typing.Any) -> None:
        
        super().__init__(**kwargs)
        # weights will be hardcoded in the source code
        # currently at arbitary values for testing
        # self.cainWeights = {"bias": 0,
        #                     "close-ghosts-count": 2.0, # cain should be near enemy ghosts
        #                     "close-food-count": 1,
        #                     "closest-food": 1.3
        # }
        self.cainWeights = {
            "bias": 0.0,
            # offense
            "dist-to-enemy-food": -4.0,      # closer to food is better
            "enemy-food-left": -1.0,         # fewer left is better
            "close-food-count": 1.5,         # more nearby food is gooder
            # safety vs defenders
            "dist-to-nearest-defender": 2.0, # farther from defenders is better
            "close-defenders-count": -3.0,   # being near defenders is bad

            # defense-related features mostly irrelevant for Cain
            "num-invaders": 0.0,
            "dist-to-nearest-invader": 0.0,
        }

        # self.abelWeights = {"bias": 0,
        #                     "close-ghosts-count": -2.0, # abel should avoid them
        #                     "close-food-count": 1,
        #                     "closest-food": 1.2}
        self.abelWeights = {
            "bias": 0.0,

            # defense
            "num-invaders": 8.0,             # invaders bad
            "dist-to-nearest-invader": -6.0, # closer to invader is better
            # optional: Abel can also prefer to be closer to invaders even when none seen (keeps him roaming)
            # keep as-is for now

            # offense features irrelevant for Abel
            "dist-to-enemy-food": 0.0,
            "enemy-food-left": 0.0,
            "close-food-count": 0.0,

            # defender features irrelevant for Abel
            "dist-to-nearest-defender": 0.0,
            "close-defenders-count": 0.0,
        }
        self.brother_index = None
        self.own_index = None

    #evaluates state for both cain and abel and returns both
    # currently a dummy version for testing
    # get a dot product of the features and the weights
    def evaluate(self, state: GameState):
        features = self.feature_extractor(state)
        #print("features: ")
        #print(features)
        #print("CainWeights: ")
        #print(self.cainWeights)
        #print("AbelWeights: ")
        #print(self.abelWeights)
        cainVal: float = 0.0
        abelVal: float = 0.0
        for f, val in features.items():
            #print("evaluting weights for %s" % (f))
            cainVal = cainVal + (val * self.cainWeights[f])
            abelVal = abelVal + (val * self.abelWeights[f])  
        return cainVal, abelVal
    
    # will need to fix weights 
    def feature_extractor(self, state: GameState) -> dict[str, float]:
        me = self.own_index
        my_pos = state.get_agent_position(me)

        if my_pos is None:
            # Can't compute distances; return safe defaults.
            return {
                "bias": 1.0,
                "num-invaders": 0.0,
                "dist-to-nearest-invader": 0.0,
                "dist-to-nearest-defender": 0.0,
                "dist-to-enemy-food": 0.0,
                "enemy-food-left": 0.0,
                "close-defenders-count": 0.0,
                "close-food-count": 0.0,
            }
        # returns a dictionary of the features extracted, whatever they're going to be
        # this is currently a dummy feature extractor copied from pacman's simple feature extractor
        # doesn't utilize the provided features library
        features: dict[str, float] = {}
        
        # Always add in a bias term.
        features['bias'] = 1.0
        distances = self._get_distances(state, self) # self is use the calling agent
        # this does make the results kind of shit for the designed purpose as there won't be consistent results for cain v abel
        # based on who is calling, but this is just for testing
        max_distance = float(state.board.width * state.board.height)
        
        me = self.own_index
        my_pos = state.get_agent_position(me)

        # enemy_states = []
        
        # for idx, st in state.get_opponent_positions().items():
        #     enemy_states.append((idx, st))


        ## find those filthy philistines
        philistine_positions = state.get_opponent_positions()  # dict[int, pos] containes all opponents
        invader_positions  = state.get_invader_positions()   # dict[int, pos] contains opponents on our side

        defenders = []
        for idx, pos in philistine_positions.items():
            if pos is None:
                continue
            if idx not in invader_positions:   # if they ain't with us...
                defenders.append((idx, pos))   # contains all opponents not on our side
                # defender_positions.appeand() # for trouble shooting, refactor to delete
        # --- Defense features ---
        features["num-invaders"] = float(len(invader_positions))
        if invader_positions:
            d = min(distances.get_distance_default(my_pos, pos, max_distance)
                    for pos in invader_positions.values()
                    if pos is not None)
            features["dist-to-nearest-invader"] = d / max_distance
        else:
            features["dist-to-nearest-invader"] = 1.0  # none seen

        # --- Offense danger features (enemy defenders) ---
        if defenders:
            d = min(distances.get_distance_default(my_pos, pos, max_distance)
                    for (_, pos) in defenders
                    if pos is not None)
            features["dist-to-nearest-defender"] = d / max_distance
        else:
            features["dist-to-nearest-defender"] = 1.0

        #TODO: add scared logic here so that we're not running away from scared ghosts
        # --- Food features ---
        food = state.get_food(agent_index=me)
        if food:
            d = min(
                distances.get_distance_default(my_pos, fpos, max_distance)
                for fpos in food
            )
            features["dist-to-enemy-food"] = d / max_distance
            features["enemy-food-left"] = float(len(food)) / 50.0
        else:
            features["dist-to-enemy-food"] = 1.0
            features["enemy-food-left"] = 0.0


        # ghost_distances = [distances.get_distance_default(state.get_agent_position(self.own_index), position, max_distance) for position in state.get_ghost_positions().values()]
        # food_distances = [distances.get_distance_default(state.get_agent_position(self.own_index), position, max_distance) for position in state.get_food()]
        # close_ghosts = [distance for distance in ghost_distances if (distance <= CLOSE_GHOST_DISTANCE)]
        # close_food = [distance for distance in food_distances if (distance <= CLOSE_FOOD_DISTANCE)]

        # close defenders (danger for Cain)
        close_defenders = 0
        for (_, pos) in defenders:
            if distances.get_distance_default(my_pos, pos, max_distance) <= CLOSE_GHOST_DISTANCE:
                close_defenders += 1
        features["close-defenders-count"] = float(close_defenders)

        # close food (opportunity for Cain)
        food_positions = state.get_food(agent_index=me)
        close_food = 0
        for fpos in food_positions:
            if distances.get_distance_default(my_pos, fpos, max_distance) <= CLOSE_FOOD_DISTANCE:
                close_food += 1
        features["close-food-count"] = float(close_food)

        # Lower all features for better optimization.
        for (key, value) in list(features.items()):
            features[key] = value / 10.0

        return features


    # helper function copied for dummy feature extractor
    def _get_distances(self,
        state: pacai.core.gamestate.GameState,
        agent: pacai.core.agent.Agent | None = None) -> pacai.search.distance.DistancePreComputer:
        distances = None

        # If there is an agent, get precomputed distances from it.
        if (agent is not None):
            distances = agent.extra_storage.get('distances', None)

        # Compute distances if we have none.
        if (distances is None):
            distances = pacai.search.distance.DistancePreComputer()
            distances.compute(state.board)

        # Save the distances in the agent (if possible).
        if (agent is not None):
            agent.extra_storage['distances'] = distances

        return distances
    
    # to be called by a get_action when own indexes are unknown
    def set_indexes(self, state: GameState):
        # state get agent_index is that of the calling agent
        self.own_index = state.agent_index
        # crunched the numbers if I'm right the team pairs are [0, 2] and [1, 3]
        if self.own_index == 0:
            self.brother_index = 2
        elif self.own_index == 2:
            self.brother_index = 0
        elif self.own_index == 1:
            self.brother_index = 3
        elif self.own_index == 3:
            self.brother_index = 1# should this be self.brother_index?
        else:
            print("Fatal error: set_indexes called when state.agent_index was %d" % (state.agent_index))
    
# Cain is the offensive agent
class Cain(Eve):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        # all member variables are inherited from Eve
    
    # returns best action, picking randomly if multiple. Decides which action is best based on CainVal
    def get_action(self, state: GameState) -> Action:
        # only on first call a game, set up own and brother indexes
        if(self.own_index == None):
            self.set_indexes(state)
        print("Cain: own_index = %d brother_index = %d" % (self.own_index, self.brother_index))
        
        # get action will act as the top layer for the tree, since return types differ
        actions = state.get_legal_actions()
        bestScore = float('-inf')
        bestActions: list[Action] = []
        for a in actions:
            print("Cain: evaluating action %s" % (str(a)))
            new_state = state.generate_successor(a)
            Cainscore, _ = self.tree(new_state, 1) # call first with depth 1, this function is essentialy depth 0
            if Cainscore == bestScore:
                # add to bestActions
                bestActions.append(a)
            elif Cainscore > bestScore:
                # reset bestActions and update bestScore
                bestScore = Cainscore
                bestActions.clear()
                bestActions.append(a)

        # now that all actions have been evaluated, return one (use rng if need to decide between them)
        if len(bestActions) == 0:
            print("Cain: Fatal Error: No action was found")
            return Action("STOP")
        else:
            action = self.rng.choice(bestActions)
            print("Cain: Chosen action %s" % str(action))
            return action

    # eval order is cainVal, abelVal
    # will be called by get_action, giving it states after an action Cain is considering
    # so it works recursively, will be returning both a cainScore and abelScore every depth
    # when someone dies they immediately respawn, essentially being teleported. 
    # so this is written that there are three turns between each agent making its own choice
    # opponents use expectimax, brother is assumed to do what gives them the best result
    # (but even that is an approximation because when brother actually goes he will look more turns in the future)
    # code is also currently written to only be able to one round into the future. MAX_DEPTH is still a constant because its good practice
    # but just adjusting that would only work if the total number of agents changed
    def tree(self, state: GameState, depth) -> tuple[float, float]:
        print("Cain: tree called at depth %d with agent_index %d" % (depth, state.agent_index))
        # get legal actions for which agent it is currently the turn of
        
        # avoid valueError(agent_index < 0)
        # this happens when there's a game over, but i have no idea why game overs are happening
        # just evaluate the state and return that
        if(state.agent_index < 0):
            # always keep this statement uncommented so we can try to track down the cause
            print("WARNING: Cain: Agent index %d did %s in depth %d resulting in state.agent_index = -1" % 
                  (state.last_agent_index, str(state.get_last_agent_action(state.last_agent_index)), (depth - 1)))
            cainVal, abelVal = self.evaluate(state)
            return cainVal, abelVal
        else:
            actions = state.get_legal_actions()
        # score keeping variables. have to initalize all of them regardless for scope reasons, but only the relevant ones will be used
        bestAbel = float('-inf')
        bestCain: list[float] = [] # has to be a list because what if there are multiple states with same bestAbel?
        # if this does happen average the Cain score (repersenting random chance of choosing those actions)
        scoreTuples: list[tuple[float, float]] = []
        for a in actions:
            print("Cain depth %d, index %d: evaluating action %s" % (depth, state.agent_index, str(a)))
            new_state = state.generate_successor(a)
            # score the state either through recursion or calling eval if we're at base case
            if depth == MAX_DEPTH:
                cainVal, abelVal = self.evaluate(new_state)
            else:
                cainVal, abelVal = self.tree(new_state, depth + 1)
            
            if state.agent_index == self.brother_index:
                # if this is Abel decide who to send up based on abelVal
                if abelVal == bestAbel:
                    bestCain.append(cainVal)
                elif abelVal > bestAbel:
                    bestAbel = abelVal
                    bestCain.clear()
                    bestCain.append(cainVal)
            elif state.agent_index == self.own_index:
                print("Cain: Fatal Error: tree() called with state where Cain was current agent")
                return 0.0, 0.0
            else: # if it is the opponent, append scores to list for future expectimaxing
                scoreTuples.append((cainVal, abelVal))

        # afterwards average bestCain if needed, and send up bestAbel, but thats to simplify the code latter
        # (since its simipler if opponent doesn't have to care if its before or after Abel)
        if state.agent_index == self.brother_index:
            bestCainAv = sum(bestCain) / len(bestCain)
            return bestCainAv, bestAbel
        else: # error check for somehow Cain being current agent already occured, unessecary here
            # currently all actions from opponents are assumed to have equal likelyhoods, so expectimax is calcuated as simple average
            options = len(scoreTuples)
            cainTotal: float = 0.0
            abelTotal: float = 0.0
            # because its tuples can't just use sum()
            for i in scoreTuples:
                cain, abel = i
                cainTotal = cainTotal + cain
                abelTotal = abelTotal + abel
            # after summing time to divide
            cainAv = cainTotal / options
            abelAv = abelTotal / options
            return cainAv, abelAv

        



# Abel is the defensive agent
class Abel(Eve):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        # all member variables come from eve

    # returns best action, picking randomly if multiple. Decides which action is best based on abelScore
    def get_action(self, state: GameState) -> Action:
        # only on first call a game, set up own and brother indexes
        if(self.own_index == None):
            self.set_indexes(state)
        print("Abel: own_index = %d brother_index = %d" % (self.own_index, self.brother_index))

        
        # get action will act as the top layer for the tree, since return types differ
        actions = state.get_legal_actions()
        bestScore = float('-inf')
        bestActions: list[Action] = []
        for a in actions:
            print("Abel: evaluating action %s" % (str(a)))
            new_state = state.generate_successor(a)
            _, abelScore = self.tree(new_state, 1) # call first with depth 1, this function is essentialy depth 0
            if abelScore == bestScore:
                # add to bestActions
                bestActions.append(a)
            elif abelScore > bestScore:
                # reset bestActions and update bestScore
                bestScore = abelScore
                bestActions.clear()
                bestActions.append(a)

        # now that all actions have been evaluated, return one (use rng if need to decide between them)
        if len(bestActions) == 0:
            print("Abel: Fatal Error: No action was found")
            return Action("STOP")
        else:
            action = self.rng.choice(bestActions)
            print("Abel: Chosen action %s" % str(action))
            return action
        
    # order for eval and tree return is cainVal, abelVal
    # get_action is top node of the tree
    # see the almost identical version of this code in Cain for more detailed comments
    def tree(self, state: GameState, depth) -> tuple[float, float]:
        print("Abel: tree called at depth %d with agent_index %d" % (depth, state.agent_index))
        # get legal actions for which agent it is currently the turn of
        # avoid valueError(agent_index < 0)
        # this happens when the game has ended, but I have no idea why this is happening so soon
        if(state.agent_index < 0):
            # always keep this statement uncommented so we can try to track down the cause
            print("WARNING: Abel: Agent index %d did %s in depth %d resulting in state.agent_index = -1" % 
                  (state.last_agent_index, str(state.get_last_agent_action(state.last_agent_index)), (depth - 1)))
            # just eval and return that value for this node
            cainVal, abelVal = self.evaluate(state)
            return cainVal, abelVal
        else:
            actions = state.get_legal_actions()
        # score keeping variables. have to initalize all of them regardless for scope reasons, but only the relevant ones will be used
        bestCain = float('-inf')
        bestAbel: list[float] = [] # has to be a list because what if there are multiple states with same bestCain?
        # if this does happen pass up average the Abel score (repersenting random chance of choosing those actions)
        scoreTuples: list[tuple[float, float]] = []
        for a in actions:
            print("Abel depth %d, index %d: evaluating action %s" % (depth, state.agent_index, str(a)))
            new_state = state.generate_successor(a)
            # score the state either through recursion or calling eval if we're at base case
            if depth == MAX_DEPTH:
                cainVal, abelVal = self.evaluate(new_state)
            else:
                cainVal, abelVal = self.tree(new_state, depth + 1)
            
            if state.agent_index == self.brother_index:
                # if this is Abel decide who to send up based on abelVal
                if cainVal == bestCain:
                    bestAbel.append(abelVal)
                elif cainVal > bestCain:
                    bestCain = cainVal
                    bestAbel.clear()
                    bestAbel.append(abelVal)
            elif state.agent_index == self.own_index:
                print("Abel: Fatal Error: tree() called with state where Abel was current agent")
                return 0.0, 0.0
            else: # if it is the opponent, append scores to list for future expectimaxing
                scoreTuples.append((cainVal, abelVal))

        # afterwards average bestCain if needed, and send up bestAbel, but thats to simplify the code latter
        # (since its simipler if opponent doesn't have to care if its before or after Abel)
        if state.agent_index == self.brother_index:
            bestAbelAv = sum(bestAbel) / len(bestAbel)
            return bestAbelAv, bestCain
        else: # error check for somehow Cain being current agent already occured, unessecary here
            # currently all actions from opponents are assumed to have equal likelyhoods, so expectimax is calcuated as simple average
            options = len(scoreTuples)
            cainTotal: float = 0.0
            abelTotal: float = 0.0
            # because its tuples can't just use sum()
            for i in scoreTuples:
                cain, abel = i
                cainTotal = cainTotal + cain
                abelTotal = abelTotal + abel
            # after summing time to divide
            cainAv = cainTotal / options
            abelAv = abelTotal / options
            return cainAv, abelAv
