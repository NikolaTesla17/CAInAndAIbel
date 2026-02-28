#by Absalom Ranelletti
#
#
# Winter 2026
from CAInAndAIbel.pacai.core.action import Action
from CAInAndAIbel.pacai.capture.gamestate import GameState
import pacai.core.agentinfo
import pacai.capture.team
import pacai.util.alias
import pacai.core.agent
import typing


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    # return pacai.capture.team.create_team_dummy()
    
    agent1_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)
    agent2_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)

    return [agent1_info, agent2_info]

#skeleton parent to Cain and Abel
class Eve(pacai.core.agent.Agent):
    def __init__(self,
            **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        # weights will be hardcoded in the source code
        self.cainWeights = {}
        self.abelWeights = {}

    #evaluates state for both cain and abel and returns both
    def evaluate(self, state: GameState):
        features = self.feature_extractor(state)
        cainVal: float = 0.0
        abelVal: float = 0.0
        return cainVal, abelVal
    
    def feature_extractor(self, state: GameState):
        # returns a dictionary of the features extracted, whatever they're going to be
        pass
    
# Cain is the offensive agent
class Cain(Eve):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        # doesn't need to know the weights
        # needs to find a way to figure out what the index of Abel is 
        self.brother_index = None
    
    def get_action(self, state: GameState) -> Action:
        # run a tree evaluating the different actions, maxing based on their cainScore
        pass

    def tree(self, state) -> pacai.core.action.Action

# Abel is the defensive agent
class Abel(Eve):
    pass