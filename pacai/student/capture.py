# by Absalom Ranelletti
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
# import sys
# ok the way WeightDict and FeatureDict work is that they're just type aliases. But I have to use this to be able to reuse the code
# about packing/unpacking training from pa3
import pacai.core.features as Feature
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

# skeleton parent to Cain and Abel


class Eve(pacai.core.agent.Agent):
    def __init__(self,
                 **kwargs: typing.Any) -> None:

        super().__init__(**kwargs)
        # weights are now held by the specific agent
        self.brother_index = None
        self.own_index = None
        # a bigger score is better. bad things have negative weights.
        # if something is universally bad but an agents actions can't effect it (like a dead ally), it isn't included
        self.weights: Feature.WeightDict = {}  # empty dict to be overriden

    # evaluate states state based on the specific weights for the relevant agent, which also has a corresponding features extractor function
    # now only returns a single float, which is that agent's perspective
    def evaluate(self, state: GameState) -> float:
        features: Feature.FeatureDict = self.feature_extractor(state)
        # print("features: ")
        print(features)
        # print("Weights: ")
        print(self.weights)
        Total: float = 0.0
        for f, val in features.items():
            # print("evaluting weights for %s" % (f))
            # if self.weights[f] == float('nan'):  # for tracking down a bug. comment out when fixed
            #   print("missing a weight")
            #   sys.exit()
            Total = Total + (val * self.weights[f])
            # cain and able have different keys in their dict, but they should be the same as the features their feature_extractor returns
        return Total

    # extracts the relevant features from state
    # each agent has their own defintion of agent, this feature is supposed to be overriden
    def feature_extractor(self, state: GameState) -> Feature.FeatureDict:
        print("Error: somehow the generic feature_extractor has been called")
        return {}
        
    # to be called by a get_action when own indexes are unknown
    def set_indexes(self, state: GameState):
        # state get agent_index is that of the calling agent
        self.own_index = state.agent_index
        # crunched the numbers if I'm right the team pairs are [0, 2] and [1,
        # 3]
        if self.own_index == 0:
            self.brother_index = 2
        elif self.own_index == 2:
            self.brother_index = 0
        elif self.own_index == 1:
            self.brother_index = 3
        elif self.own_index == 3:
            self.brother_index = 1
        else:
            print(
                "Fatal error: set_indexes called when state.agent_index was %d" %
                (state.agent_index))
            
    # in this build get_action is the same for both of the brothers, with their eval and feature extractors being different
    # returns best action, picking randomly if multiple. Decides which action is the best based on its personal
    # evaluate function
    # this one just happens to be copied from Cain so thats why it has the debug prints it does
    def get_action(self, state: GameState) -> Action:
        # only on first call a game, set up own and brother indexes
        if (self.own_index is None):
            self.set_indexes(state)
        # print("%s: own_index = %d brother_index = %d" % (str(isinstance(self, Cain)), self.own_index, self.brother_index))

        # get action will act as the top layer for the tree, since return types
        # differ
        # below is trying to stop STOP from running all the time
        actions = state.get_legal_actions()
        # actions = [a for a in state.get_legal_actions() if str(a) != 'STOP']
        # if not actions:
        #     actions = state.get_legal_actions()
        # print("Agent %d all actions: %s" % (self.own_index, str(actions)))
        bestScore = float('-inf')
        bestActions: list[Action] = []
        for a in actions:
            new_state = state.generate_successor(a)
            # call first with depth 1, this function is essentialy depth 0
            score = self.tree(new_state, 1)
            # print("evaluated action %s to have score %f" % (str(a), score))
            if score == bestScore:
                # print("this is equal to bestScore(%f), adding to best actions" % (bestScore))
                # add to bestActions
                bestActions.append(a)
                # print("bestActions: %s" % (str(bestActions)))
            elif score > bestScore:
                # reset bestActions and update bestScore
                bestScore = score
                bestActions.clear()
                bestActions.append(a)
                # print("this is better than bestScore. New best value is %f" % (bestScore))
                # print("bestActions is now: %s" % (str(bestActions)))

        # now that all actions have been evaluated, return one (use rng if need
        # to decide between them)
        if len(bestActions) == 0:
            print("Agent %d: Fatal Error: No action was found" % (self.own_index))
            # sys.exit()
            # for getting past the auto grader if i keep crashing, but I don't think that's happening
            return Action("STOP")
        else:
            # print("Cain: best actions %s" % str(bestActions))
            action = self.rng.choice(bestActions)
            # print("Cain: Chosen action %s" % str(action))
            return action
    
    # is called by get_action. Tree is passed a state after the action it is being considered
    # for time savings, the tree is calucated as if the brother just does STOP. Each brother has his own evaluate and feature extractor.
    # because of this, it is essentially just an expectimax tree because its just thinking about the next turns the opponents are going to take
    # this just happened to be copied from Cain, hence the debug print comments
    def tree(self, state: GameState, depth) -> float:
        # print("Tree called at depth %d with agent_index %d" % (depth, state.agent_index))
        # get legal actions for which agent it is currently the turn of
        if state.agent_index < 0:
            return self.evaluate(state)
        actions = state.get_legal_actions()
        if not actions:
            return self.evaluate(state)

        if state.agent_index == self.own_index:  # i think determining the agent type upfront should speed things up'
            bestVal = float('-inf')
            # print("Agent %d: Non-fatal error: An agent is somehow having a turn in its own tree. That's not supposed to happen" % (self.own_index))
            # it seems like sometimes this will happen??? so just leave it as is
            # print(str(state.get_agent_indexes()))
            for a in actions:
                # print("Cain depth %d, index %d: evaluating action %s" % (depth, state.agent_index, str(a)))
                new_state = state.generate_successor(a)
                # score the state either through recursion or calling eval if we're
                # at base case
                if depth >= MAX_DEPTH:
                    Val = self.evaluate(new_state)
                else:
                    Val = self.tree(new_state, depth + 1)

                if Val > bestVal:
                    bestVal = Val
            return bestVal
        elif state.agent_index == self.brother_index:
            # brother only generates one state
            new_state = state.generate_successor(Action("STOP"))
            # print("succesfully applied stop. new agent index = %d" % (new_state.agent_index))
            if depth >= MAX_DEPTH:
                Val = self.evaluate(new_state)
            else:
                Val = self.tree(new_state, depth + 1)
            return Val
        else:  # error check for somehow calling agent being current agent already occured, unessecary here
            # currently all actions from opponents are assumed to have equal
            # likelyhoods, so expectimax is calcuated as simple average
            # options = len(scoreTuples)
            Total: float = 0.0
            for a in actions:
                new_state = state.generate_successor(a)
                if depth >= MAX_DEPTH:
                    Val = self.evaluate(new_state)
                else:
                    Val = self.tree(new_state, depth + 1)
                Total += Val
            n = len(actions)
            # hey if agent is crashing it might be because divide by zero is happening here. very unlikely though, so I didn't
            # write an error catch so pipelining can be more effective, since speed is extremely important rn
            return Total / n

# i'm pretty sure comput gets the actual distances by traversing the board, so there's nowhere to impliment bfs
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


# Cain is the offensive agent
class Cain(Eve):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        # override weights
        self.weights: Feature.WeightDict = {
            "cain-dist-to-enemy-food": -4.0,      # closer to food is better
            "enemy-food-left": -1.0,         # fewer left is better
            "cain-close-food-count": 1.5,         # more nearby food is gooder
            # safety vs defenders
            # farther from defenders is better
            "cain-dist-to-nearest-nonscared-defender": 2.0,
            "cain-dist-to-nearest-scared-defender": -2.0,  # unless they're scared
            "cain-close-nonscared-defenders-count": -3.0,   # being near defenders is bad
            "cain-close-scared-defenders-count": 3.0,  # unless they're scared
            # irrelevant features removed

            "normalized-score": 15.0,  # the actual score is extremely important
            "cain-correct-zone": 10.0,  # important to be in the correct zone
            # but it doesn't matter to cain if abel is in the correct zone or
            # not
            
            "cain-afraid": -0.3,
            # being dead is bad. but because of quick respawn isn't as
            # important as the score
            "cain-dead": -10.0,
            # seemingly causes timeout
            # "center-distance": -3.0
        }        # index member variables held by eve

    # self is the agent who is at the top of the stack calling this to decide what action to take next. Could be cain or abel
    # the agent from state is who is at the bottom of the tree, and is irrelevant
    def feature_extractor(self, state: GameState) -> Feature.FeatureDict:
        # default version of the dictionary with placeholder values so what is
        # used is easily referenced
        features: Feature.FeatureDict = {
            "cain-dist-to-nearest-nonscared-defender": 0.0,
            "cain-dist-to-nearest-scared-defender": 0.0,
            "cain-dist-to-enemy-food": 0.0,
            "enemy-food-left": 0.0,
            "cain-close-nonscared-defenders-count": 0.0,
            "cain-close-scared-defenders-count": 0.0,
            "cain-close-food-count": 0.0,
            "normalized-score": 0.0,
            "cain-correct-zone": 0.0,  # 1/0 if cain is in the enemy area
            "cain-afraid": 0.0,  # 1/0 if cain is afraid
            # these binary values are expected to have relatively high positive
            # or negative weights
            # currently seems like center-distance is the straw that
            # breaks the camels back time wise,
            # hopefully when the weights are best we won't need it
            # "center-distance": 0.0,
            # won't be calcuated unless they're in the wrong zone, otherwise its a value with negative weight to encourage
            # crossing over to the other side
            # extant scared defenders is removed because its either about how many of the opponents are currently defenders (can't control)
            # or it actually decreases as opponents are eaten. or its about time. Since optimization is so important, i've removed what
            # was always going to be a pretty inconsquential feature
        }

        # this stores precomputed distances in the memory of the agent
        distances = _get_distances(state, self)
        # unless I've really messed up my understanding, its the literal distances between the spaces and thus the agent passed is
        # only for assigning where to store this value. So it can be used with
        # the pos from both cainIndex and abelIndex
        max_distance = float(state.board.width * state.board.height)

        # own index is simply usued throughout this. Can't get pylance to stop complaining
        # get score that is always positive in our own direction
        score = state.get_normalized_score(self.own_index)
        # self.own_index has to be typed to int | None, so pylance is always going to be angry. but there should never be a situation
        # where it reaches this function as None
        features["normalized-score"] = score

        # find all opponents, divided by side and if they're scared
        # since the index parameter is just for team, can use own_index
        philistine_positions = state.get_nonscared_opponent_positions(
            self.own_index)
        scared_philistine_positions = state.get_scared_opponent_positions(
            self.own_index)
        # invader_positions = state.get_invader_positions(self.own_index) # Cain doesn't care about invaders
        # all these get_[]_positions return list of (index, pos)
        # invaders don't get scared, you do

        nonscared_defenders = []
        for idx, pos in philistine_positions.items():
            if pos is None:
                continue
            if state.is_ghost(idx):
                # without invader_positions, check to see who is on their own
                # side of the board by seeing if they're ghosts
                nonscared_defenders.append((idx, pos))

        scared_defenders = []
        for idx, pos in scared_philistine_positions.items():
            if pos is None:
                continue
            if state.is_ghost(idx):
                scared_defenders.append((idx, pos))

        # as cain, no need to extract defense features
        
        # first see if Cain is even alive before computing his relevant stats
        cainpos = state.get_agent_position(self.own_index)
        if cainpos is None:
            features["cain-dead"] = 1.0
        else:
            if state.is_scared(self.own_index):
                features["cain-afraid"] = 1.0
            if not state.is_ghost(self.own_index):
                features["cain-correct-zone"] = 1.0
            # else:
                # problem, we have no way to encourage them to leave their zone if it takes too many steps
                # until now
                # also going to have to get maze-distance working and hope that doesn't cause timeout
                # middle: int = int(state.board.width / 2)
                # columns are indexed to 1
                # encourage moving towards the middle when you're in the wrong side, don't care about height
                # features["center-distance"] = abs(middle - cainpos.col) / max_distance

            # --- Offense danger features (enemy defenders) ---
            if nonscared_defenders:
                d = min(
                    distances.get_distance_default(
                        cainpos, pos, max_distance) for (
                        _, pos) in nonscared_defenders if pos is not None)
                features["cain-dist-to-nearest-nonscared-defender"] = d / max_distance
            if scared_defenders:
                d = min(
                    distances.get_distance_default(
                        cainpos, pos, max_distance) for (
                        _, pos) in scared_defenders if pos is not None)
                features["cain-dist-to-nearest-scared-defender"] = d / max_distance
            # this is also only relevant to cain
            # --- Food features ---
            food = state.get_food(self.own_index)
            if food:
                d = min(
                    distances.get_distance_default(cainpos, fpos, 0.0)
                    for fpos in food
                )
                features["cain-dist-to-enemy-food"] = d / max_distance
                # not sure what this extra division here is for, but I'll leave
                # it
                features["enemy-food-left"] = float(len(food)) / 50

            # close defenders (danger (or opportuinity if they're scared!) for
            # Cain)
            close_nonscared_defenders = 0
            for (_, pos) in nonscared_defenders:
                if distances.get_distance_default(
                        cainpos, pos, max_distance) <= CLOSE_GHOST_DISTANCE:
                    close_nonscared_defenders += 1
            features["cain-close-nonscared-defenders-count"] = float(
                close_nonscared_defenders)
            close_scared_defenders = 0
            for (_, pos) in scared_defenders:
                if distances.get_distance_default(
                        cainpos, pos, max_distance) <= CLOSE_GHOST_DISTANCE:
                    close_scared_defenders += 1
            features["cain-close-scared-defenders-count"] = float(
                close_scared_defenders)

            # close food (opportunity for Cain)
            food_positions = state.get_food(self.own_index)
            close_food = 0
            for fpos in food_positions:
                if distances.get_distance_default(
                        cainpos, fpos, max_distance) <= CLOSE_FOOD_DISTANCE:
                    close_food += 1
            features["cain-close-food-count"] = float(close_food)

        # thing from the dummy extractor I don't understand, but we need all the optimization we can get
        # ""Lower all features for better optimization.""
        for (key, value) in list(features.items()):
            features[key] = value / 10.0

        return features

# Abel is the defensive agent
class Abel(Eve):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        # now weights are only held by the relevant agent
        self.weights: Feature.FeatureDict = {
            "num-invaders": -1.0,             # invaders bad
            "abel-dist-to-nearest-nonscared-invader": -6.0,  # closer to invader is better
            "abel-dist-to-nearest-scared-invader": 6.0,  # unless abel is scared
            "abel-close-nonscared-invaders-count": 3.0,  # same idea with these values
            "abel-close-scared-invaders-count": -3.0,
            # removed weights irrelevant to Abel

            "normalized-score": 15.0,  # score extremely important
            "abel-correct-zone": 10.0,  # 1/0 if abel is in the enemy area
            "abel-afraid": -3.0,  # as a defender, abel cares much more about being afraid
            "abel-dead": -10.0,  # 1/0 if abel is dead
            # "center-distance": -3.0
        }
        # own/brother index memeber variables held by Eve

    # self is the agent who is at the top of the stack calling this to decide what action to take next.
    # the agent from state is who is at the bottom of the tree, and is irrelevant
    def feature_extractor(self, state: GameState) -> Feature.FeatureDict:
        # default version of the dictionary with placeholder values so what is
        # used is easily referenced
        features: Feature.FeatureDict = {
            "num-invaders": 0.0,
            # slight misnomer, "scared" here refers to abel, not the invader
            "abel-dist-to-nearest-nonscared-invader": 0.0,
            # we're not just * -1 them in case it turns out they would be best
            # with different weights
            "abel-dist-to-nearest-scared-invader": 0.0,
            "abel-close-nonscared-invaders-count": 0.0,
            "abel-close-scared-invaders-count": 0.0,
            "normalized-score": 0.0,
            "abel-correct-zone": 0.0,  # 1/0 if abel is in the enemy area
            "abel-afraid": 0.0,  # 1/0 if abel is afraid
            "abel-dead": 0.0,  # 1/0 if abel is dead
            # these binary values are expected to have relatively high positive
            # or negative weights
            # "center-distance": 0.0,
        }

        # this stores precomputed distances in the memory of the agent
        distances = _get_distances(state, self)
        # unless I've really messed up my understanding, its the literal distances between the spaces and thus the agent passed is
        # only for assigning where to store this value. So it can be used with
        # the pos from both cainIndex and abelIndex
        max_distance = float(state.board.width * state.board.height)

        # only self.own_index matters, this is the abel specific feature extractor

        # get score that is always positive in our own direction
        score = state.get_normalized_score(self.own_index)
        # self.own_index has to be typed to int | None, so pylance is always going to be angry. but there should never be a situation
        # where it reaches this function as None
        features["normalized-score"] = score
        # to abel, only invaders matter
        invader_positions = state.get_invader_positions(self.own_index)
        # all these get_[]_positions return list of (index, pos)
        # invaders don't get scared, you do

        # before calcuating defense features, see if Abel is even alive
        abelpos = state.get_agent_position(self.own_index)
        if abelpos is None:
            features["abel-dead"] = 1.0
            # if this is the case, the abel related features don't have values assigned to them
            # their default values of 0.0 are fine
        else:
            if state.is_scared(self.own_index):
                features["abel-afraid"] = 1.0
                # leaving the binary features to their default value for the
                # "no" answer is fine, as they're already 0.0
            if state.is_ghost(self.own_index):
                features["abel-correct-zone"] = 1.0
            # else:
                # problem, we have no way to encourage them to leave their zone if it takes too many steps
                # until now
                # also going to have to get maze-distance working and hope that doesn't cause timeout
                # middle: int = int(state.board.width / 2)
                # encourage moving towards the middle when you're in the wrong side, don't care about height
                # features["center-distance"] = abs(middle - abelpos.col) / max_distance
            # --- Defense features ---
            features["num-invaders"] = float(len(invader_positions))
            if invader_positions:
                d = min(distances.get_distance_default(abelpos, pos, 0.0)
                        for pos in invader_positions.values() if pos is not None)
                # assign to the apporpirate feature. irrelevant features are
                # already assigned to 0.0
                if state.is_scared(self.own_index):
                    features["abel-dist-to-nearest-scared-invader"] = d / max_distance
                    # I don't know why you'd do this, but for some reason its what works and prevents timing out
                else:
                    features["abel-dist-to-nearest-nonscared-invader"] = d / max_distance

                # close invaders, either good or bad depending on if abel is
                # currently scared, so they exist differently
                close_invaders = 0
                for _, pos in invader_positions.items():
                    if distances.get_distance_default(
                            abelpos, pos, max_distance) <= CLOSE_GHOST_DISTANCE:
                        close_invaders += 1
                if (state.is_scared(self.own_index)):
                    features["abel-close-scared-invaders-count"] = float(
                        close_invaders)
                else:
                    features["abel-close-nonscared-invaders-count"] = float(
                        close_invaders)

        # cain irrelevant
        # don't know what this does, but comments says it increases optimization
        # ""Lower all features for better optimization.""
        for (key, value) in list(features.items()):
            features[key] = value / 10.0

        return features