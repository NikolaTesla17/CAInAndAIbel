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
        # Since there seems to be some confusion, the higher the evaluate score the better
        # remember, even if something is bad overall, if the agent can't make a decision to effect it doesn't matter to their evaluation
        # the double evaluation is just to allow it to make a more accurate prediction of the future
        self.cainWeights = {
            # offense
            "cain-dist-to-enemy-food": -4.0,      # closer to food is better
            "enemy-food-left": -1.0,         # fewer left is better
            "cain-close-food-count": 1.5,         # more nearby food is gooder
            # safety vs defenders
            "cain-dist-to-nearest-nonscared-defender": 2.0, # farther from defenders is better
            "cain-dist-to-nearest-scared-defender": -2.0, # unless they're scared
            "cain-close-nonscared-defenders-count": -3.0,   # being near defenders is bad
            "cain-close-scared-defenders-count": 3.0, # unless they're scared
            "scared-defenders-extant": 1.2, # all extant scared defenders
            # defense irrelevant for Cain
            "num-invaders": 0.0,
            "dist-to-nearest-invader": 0.0,
            "abel-dist-to-nearest-nonscared-invader": 0.0,
            "abel-dist-to-nearest-scared-invader": 0.0,
            "abel-close-nonscared-invaders-count": 0.0,
            "abel-close-scared-invaders-count": 0.0,

            "normalized-score": 15.0, # the actual score is extremely important 
            "cain-correct-zone": 10.0, # important to be in the correct zone
            "abel-correct-zone": 0.0, # but it doesn't matter to cain if abel is in the correct zone or not
            "abel-afraid": 0.0, # same if he's afraid
            "cain-afraid": -0.3, # cain shouldn't care all that much if he should be afraid because that doesn't effect pacmen
            "abel-dead": 0.0,
            "cain-dead": -10.0, # being dead is bad. but because of quick respawn isn't as important as the score
        }

        self.abelWeights = {
            # defense
            "num-invaders": -1.0,             # invaders bad
            "abel-dist-to-nearest-nonscared-invader": -6.0, # closer to invader is better
            "abel-dist-to-nearest-scared-invader": 6.0, # unless abel is scared
            "abel-close-nonscared-invaders-count": 3.0, # same idea with these values
            "abel-close-scared-invaders-count": -3.0,
            # offense irrelevant for Abel
            "cain-dist-to-nearest-nonscared-defender": 0.0,
            "cain-dist-to-nearest-scared-defender": 0.0,
            "cain-dist-to-enemy-food": 0.0,
            "enemy-food-left": 0.0,
            "cain-close-nonscared-defenders-count": 0.0,
            "cain-close-scared-defenders-count": 0.0,
            "scared-defenders-extant": 0.0,
            "cain-close-food-count": 0.0,

            "normalized-score": 15.0, # score extremely important
            "cain-correct-zone": 0.0, # 1/0 if cain is in the enemy area
            "abel-correct-zone": 10.0, # 1/0 if abel is in the enemy area
            "abel-afraid": -3.0, # as a defender, abel cares much more about being afraid
            "cain-afraid": 0.0, # 1/0 if cain is afraid
            "abel-dead": -10.0, #1/0 if abel is dead
            "cain-dead": 0.0, #1/0 if cain is dead
                
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
    
    # extracts the relevant features from state
    # self is the agent who is at the top of the stack calling this to decide what action to take next. Could be cain or abel
    # the agent from state is who is at the bottom of the tree, and is irrelevant
    def feature_extractor(self, state: GameState) -> dict[str, float]:
        # default version of the dictionary with placeholder values so what is used is easily referenced
        features: dict[str, float] = {
                "num-invaders": 0.0,
                "abel-dist-to-nearest-nonscared-invader": 0.0, # slight misnomer, "scared" here refers to abel, not the invader
                "abel-dist-to-nearest-scared-invader": 0.0, # we're not just * -1 them in case it turns out they would be best with different weights
                "cain-dist-to-nearest-nonscared-defender": 0.0,
                "cain-dist-to-nearest-scared-defender": 0.0,
                "cain-dist-to-enemy-food": 0.0,
                "enemy-food-left": 0.0,
                "cain-close-nonscared-defenders-count": 0.0,
                "cain-close-scared-defenders-count": 0.0,
                "abel-close-nonscared-invaders-count": 0.0,
                "abel-close-scared-invaders-count": 0.0,
                "cain-close-food-count": 0.0,
                "normalized-score": 0.0,
                "cain-correct-zone": 0.0, # 1/0 if cain is in the enemy area
                "abel-correct-zone": 0.0, # 1/0 if abel is in the enemy area
                "abel-afraid": 0.0, # 1/0 if abel is afraid
                "cain-afraid": 0.0, # 1/0 if cain is afraid
                "abel-dead": 0.0, #1/0 if abel is dead
                "cain-dead": 0.0, #1/0 if cain is dead
                "scared-defenders-extant": 0.0, # all extant scared defenders
                # these binary values are expected to have relatively high positive or negative weights
        }

        distances = _get_distances(state, self) # this stores precomputed distances in the memory of the agent
        # unless I've really messed up my understanding, its the literal distances between the spaces and thus the agent passed is
        # only for assigning where to store this value. So it can be used with the pos from both cainIndex and abelIndex
        max_distance = float(state.board.width * state.board.height)
        
        # figure out the indexes of Cain and Abel, regardless of who calls
        cainIndex: int | None = None
        abelIndex: int |None = None
        if(isinstance(self, Cain)):
            cainIndex = self.own_index
            abelIndex = self.brother_index
        else:
            cainIndex = self.brother_index
            abelIndex = self.own_index

        score = state.get_normalized_score(self.own_index) # get score that is always positive in our own direction
        # self.own_index has to be typed to int | None, so pylance is always going to be angry. but there should never be a situation
        # where it reaches this function as None
        features["score"] = score

        # find all opponents, divided by side and if they're scared
        # since the index parameter is just for team, can use own_index
        philistine_positions = state.get_nonscared_opponent_positions(self.own_index)
        scared_philistine_positions = state.get_scared_opponent_positions(self.own_index)
        invader_positions  = state.get_invader_positions(self.own_index)
        # all these get_[]_positions return list of (index, pos)
        # invaders don't get scared, you do

        nonscared_defenders = []
        for idx, pos in philistine_positions.items():
            if pos is None:
                continue
            if idx not in invader_positions and state.is_ghost(idx):
                nonscared_defenders.append((idx, pos))

        scared_defenders = []
        for idx, pos in scared_philistine_positions.items():
            if pos is None:
                continue
            if idx not in invader_positions and state.is_ghost(idx):
                scared_defenders.append((idx, pos))

        # before calcuating defense features, see if Abel is even alive
        abelpos = state.get_agent_position(abelIndex)
        if abelpos is None:
            features["abel-dead"] = 1.0
            # if this is the case, the abel related features don't have values assigned to them
            # their default values of 0.0 are fine
        else:
            if state.is_scared(abelIndex):
                features["abel-afraid"] = 1.0
                # leaving the binary features to their default value for the "no" answer is fine, as they're already 0.0
            if state.is_ghost(abelIndex):
                features["abel-correct-zone"] = 1.0
            # --- Defense features ---
            features["num-invaders"] = float(len(invader_positions))
            if invader_positions:
                d = min(distances.get_distance_default(abelpos, pos, max_distance)
                    for pos in invader_positions.values()
                    if pos is not None)
                # assign to the apporpirate feature. irrelevant features are already assigned to 0.0
                if state.is_scared(abelIndex):
                    features["abel-dist-to-nearest-scared-invader"] = d / max_distance
                else:
                    features["abel-dist-to-nearest-nonscared-invader"] = d / max_distance
                
                # close invaders, either good or bad depending on if abel is currently scared, so they exist differently
                close_invaders = 0
                for _, pos in invader_positions.items():
                    if distances.get_distance_default(abelpos, pos, max_distance) <= CLOSE_GHOST_DISTANCE:
                        close_invaders += 1
                if(state.is_scared(abelIndex)):
                    features["abel-close-scared-invaders-count"] = float(close_invaders)
                else:
                    features["abel-close-nonscared-invaders-count"] = float(close_invaders)



        # first see if Cain is even alive before computing his relevant stats
        cainpos = state.get_agent_position(cainIndex)
        if cainpos is None:
            features["cain-dead"] = 1.0
        else:
            if state.is_scared(cainIndex):
                features["cain-afraid"] = 1.0
            if state.is_ghost(cainIndex) == False:
                features["cain-correct-zone"] = 1.0
            # --- Offense danger features (enemy defenders) ---
            if nonscared_defenders:
                d = min(distances.get_distance_default(cainpos, pos, max_distance)
                        for (_, pos) in nonscared_defenders
                        if pos is not None)
                features["cain-dist-to-nearest-nonscared-defender"] = d / max_distance
            # the reason for this logic being different is on defender there's only one abel to be scared, but if a ghost dies before the
            # scared timer runs out it will respawn not scared
            features["scared-defenders-extant"] = float(len(scared_defenders) > 0)
            if scared_defenders:
                d = min(distances.get_distance_default(cainpos, pos, max_distance)
                        for (_, pos) in scared_defenders
                        if pos is not None)
                features["cain-dist-to-nearest-scared-defender"] = d / max_distance
            # this is also only relevant to cain
            # --- Food features ---
            food = state.get_food(cainIndex)
            if food:
                d = min(
                    distances.get_distance_default(cainpos, fpos, max_distance)
                    for fpos in food
                )
                features["cain-dist-to-enemy-food"] = d / max_distance
                features["enemy-food-left"] = float(len(food)) / 50.0 # not sure what this extra division here is for, but I'll leave it


            # close defenders (danger (or opportuinity if they're scared!) for Cain)
            close_nonscared_defenders = 0
            for (_, pos) in nonscared_defenders:
                if distances.get_distance_default(cainpos, pos, max_distance) <= CLOSE_GHOST_DISTANCE:
                    close_nonscared_defenders += 1
            features["cain-close-nonscared-defenders-count"] = float(close_nonscared_defenders)
            close_scared_defenders = 0
            for (_, pos) in scared_defenders:
                if distances.get_distance_default(cainpos, pos, max_distance) <= CLOSE_GHOST_DISTANCE:
                    close_scared_defenders += 1
            features["cain-close-nonscared-defenders-count"] = float(close_scared_defenders)

            # close food (opportunity for Cain)
            food_positions = state.get_food(cainIndex)
            close_food = 0
            for fpos in food_positions:
                if distances.get_distance_default(cainpos, fpos, max_distance) <= CLOSE_FOOD_DISTANCE:
                    close_food += 1
            features["cain-close-food-count"] = float(close_food)

        # thing from the dummy extractor I don't think we want to use, but I'm keeping it here just in case it turns out we do
        # ""Lower all features for better optimization.""
        #for (key, value) in list(features.items()):
            #features[key] = value / 10.0

        return features

def _get_distances(
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
        #print("Cain: own_index = %d brother_index = %d" % (self.own_index, self.brother_index))
        
        # get action will act as the top layer for the tree, since return types differ
        actions = state.get_legal_actions()
        print("abel all actions: %s" % (str(actions)))
        bestScore = float('-inf')
        bestActions: list[Action] = []
        for a in actions:
            
            new_state = state.generate_successor(a)
            Cainscore, _ = self.tree(new_state, 1) # call first with depth 1, this function is essentialy depth 0
            print("Cain: evaluated action %s to have score %f" % (str(a), Cainscore))
            if Cainscore == bestScore:
                # print("this is equal to bestScore(%f), adding to best actions" % (bestScore))
                # add to bestActions
                bestActions.append(a)
                # print("bestActions: %s" % (str(bestActions)))
            elif Cainscore > bestScore:
                # reset bestActions and update bestScore
                bestScore = Cainscore
                bestActions.clear()
                bestActions.append(a)
                # print("this is better than bestScore. New best value is %f" % (bestScore))
                # print("bestActions is now: %s" % (str(bestActions)))

        # now that all actions have been evaluated, return one (use rng if need to decide between them)
        if len(bestActions) == 0:
            print("Cain: Fatal Error: No action was found")
            return Action("STOP")
        else:
            # print("Cain: best actions %s" % str(bestActions))
            action = self.rng.choice(bestActions)
            # print("Cain: Chosen action %s" % str(action))
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
        # print("Cain: tree called at depth %d with agent_index %d" % (depth, state.agent_index))
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
            # print("Cain depth %d, index %d: evaluating action %s" % (depth, state.agent_index, str(a)))
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
        # print("Abel: own_index = %d brother_index = %d" % (self.own_index, self.brother_index))

        
        # get action will act as the top layer for the tree, since return types differ
        actions = state.get_legal_actions()
        bestScore = float('-inf')
        bestActions: list[Action] = []
        print("abel all actions: %s" % (str(actions)))
        for a in actions:
           # print("Abel: evaluating action %s" % (str(a)))
            new_state = state.generate_successor(a)
            _, abelScore = self.tree(new_state, 1) # call first with depth 1, this function is essentialy depth 0
            print("Cain: evaluated action %s to have score %f" % (str(a), abelScore))
            if abelScore == bestScore:
                # print("this is equal to bestScore(%f), adding to best actions" % (bestScore))
                # add to bestActions
                bestActions.append(a)
                # print("bestActions: %s" % (str(bestActions)))
            elif abelScore > bestScore:
                # reset bestActions and update bestScore
                bestScore = abelScore
                bestActions.clear()
                bestActions.append(a)
                # print("this is better than bestScore. New best value is %f" % (bestScore))
                # print("bestActions is now: %s" % (str(bestActions)))

        # now that all actions have been evaluated, return one (use rng if need to decide between them)
        if len(bestActions) == 0:
            print("Abel: Fatal Error: No action was found")
            return Action("STOP")
        else:
            print("Abel: best actions %s" % str(bestActions))
            action = self.rng.choice(bestActions)
            print("Abel: Chosen action %s" % str(action))
            return action
        
    # order for eval and tree return is cainVal, abelVal
    # get_action is top node of the tree
    # see the almost identical version of this code in Cain for more detailed comments
    def tree(self, state: GameState, depth) -> tuple[float, float]:
        # print("Abel: tree called at depth %d with agent_index %d" % (depth, state.agent_index))
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
            # print("Abel depth %d, index %d: evaluating action %s" % (depth, state.agent_index, str(a)))
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
