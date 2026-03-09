# for doing local q training of the agents of capture.py to hardcode them with better weights
# i don't know how to do the python "functions as parameters" stuff well enough to do anything fancier than making these
# two identical child functions
import pacai.student.capture as capture
import pacai.core.agentinfo
import typing
import pacai.core.features as Features
import pacai.core.gamestate
import logging
import edq.util.json
import pacai.core.board
import pacai.core.action as action
import pacai.core.agentaction
from pacai.capture.gamestate import GameState

#how this works is that alias.py has been modified to include an alias for capture-team-training. use that as an arg to run the training
def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    # return pacai.capture.team.create_team_dummy()

    agent1_info = pacai.core.agentinfo.AgentInfo(f"{__name__}.Learning_Cain")
    agent2_info = pacai.core.agentinfo.AgentInfo(f"{__name__}.Learning_Abel")

    return [agent1_info, agent2_info]

# code has primarily been copied from AR's PA3, starting with the approximate q learning agent and then working up its inheritance
# since this class has to be the child of Cain/Abel.
# its gonna be basically the same code in both Learning_Cain and Learning_Abel, but like I said I'm not good enough at the functions
# as parameters stuff seen throughout this code base to trust them at this late hour
class Learning_Cain(capture.Cain):
    def __init__(self,
        **kwargs: typing.Any) -> None:
        # starting weights comes from parent class
        # feature extractor is also hardcoded to parent class
        super().__init__(**kwargs)
            
        #q learning related variables
        # these values are the defaults specified in pacai/agents/mdp.py
        # these aren't controlled by arguments
        self.discount_rate = 0.9
        self.learning_rate = 0.5
        self.exploration_rate = 0.3
        self.last_state: GameState | None = None
        self.total_rewards: float = 0.0
        self.training = True # we will always be training, but just in case I'm keeping the old infastructure



    def pack_training_info(self) -> dict[str, typing.Any]:
        return {
            'weights': self.weights,
    }

    def unpack_training_info(self, data: dict[str, typing.Any]) -> None:
        self.weights = Features.WeightDict(data.get('weights', {}))

    def game_complete(self, final_state: GameState) -> None:
        super().game_complete(final_state)
        logging.debug("%s Weights: %s." % (str(self), edq.util.json.dumps(self.weights))) # very important for copying weights to capture.py

    def game_start(self, initial_state: GameState) -> None:
        self.last_state = initial_state

    # hopefully teh seperate game_complete and game_complete_full doesn't cause problems
    def game_complete_full(self,
            final_state: GameState,
            ) -> pacai.core.agentaction.AgentAction:
        if (self.training):
            logging.debug("Completed training epoch %d.", self.training_epoch)

        self.update(final_state)
        average_reward = 0.0
        num_actions = len(final_state.get_agent_actions(self.agent_index))
        if (num_actions > 0):
            average_reward = self.total_rewards / num_actions
        logging.debug("Made %d moves for a total of %0.2f rewards (average: %0.2f).",
                num_actions, self.total_rewards, average_reward)
        # Store the training information for the next epoch's agent.
        agent_action = super().game_complete_full(final_state)
        agent_action.training_info['training_info'] = self.pack_training_info()
        return agent_action

    def update(self, new_state: GameState) -> None:
        """
        Update the agent based on the difference between the old state and new state.
        """
        # Get the most recent action.
        # this is called before the new action is taken. So its the action that brought you to your current state
        last_action = new_state.get_last_agent_action(self.own_index) # new state won't be right after our agent's turn, but this
        # will still extract the right action
        if (last_action is None):
            # No action has been taken yet, don't update.
            return

        # Update the last seen state.
        old_state = self.last_state # this would be the last state where this agent took an action. new state is 3 turns later
        self.last_state = new_state.copy()

        if (old_state is None):
            # We don't have an old state to compare against yet.
            return

        # Compute and store the score delta.
        # remember to normalize it
        score_delta = new_state.get_normalized_score(self.own_index) - old_state.get_normalized_score(self.own_index)
        self.total_rewards += score_delta

        # Do not update if we are not training.
        if (not self.training):
            return

        old_position = old_state.get_agent_position(self.own_index)
        if (old_position is None):
            # The agent was not on the board the last turn. Did they respawn?
            return

        new_position = self.last_positions[-1]

        self.update_qvalue(score_delta, last_action,
            old_state, new_state,
            old_position, new_position)

    def get_action(self, state: GameState) -> action.Action:
        # Update the agent by learning from the environment.
        # This code should not change and always be the first thing done in this method.
        print("getting new action for %d" % self.agent_index)
        self.update(state)
        # use epilsion to calcuate if I do the policy action or a random one
        print("after updating that process")
        legal_actions: list[action.Action] = state.get_legal_actions()
        roll = self.rng.randrange(0, 100)
        if roll <= self.exploration_rate:
            # choose random
            a: action.Action = self.rng.choice(legal_actions)
            # print("Choosing random action " + str(a))
            return a
        else:
           # our policy is outlined in the parent classes, so just do super
           return super().get_action(state)

    def get_mdp_state_value(self, game_state: GameState) -> float:
        """Finds the maximum Q-value among all legal actions for a state."""
        # the problem with how this code was is that its evaluating it on the next step, which is the opponents or brother's turn
        # what I want is the score that comes from doing the policy action
        # I essentially need to do super().get_action again, except I return the bestVal instead of actions
        # the cheat for this is to call tree() with depth 0 (and double ignore the print about having the calling agent be in the tree being wrong)
        return super().tree(game_state, 0)

    def get_qvalue(self, old_state, new_state,
        action: action.Action) -> float:
        # even when i'm not using old_state and action it just feels weird to call get_qvalue without them
        # just run evaluate
        q = self.evaluate(new_state)
        return q

    def update_qvalue(self,
        reward: float,
        action: action.Action,
        old_game_state: pacai.core.gamestate.GameState, new_game_state: pacai.core.gamestate.GameState,
        old_position: pacai.core.board.Position | None, new_position: pacai.core.board.Position | None,
        ) -> None:
        # since weights are being remembered instead of q values, update with:
        # wi ← wi + α ∗ [correction] ∗ fi(s,a)
        # correction = (R(s,a) + γ ∗ V'(s)) − Q(s,a)

        # correction = (R(s,a) + γ ∗ V'(s)) − Q(s,a)
        # since its s and not s', use the old state
        correction = 0.0
        if (new_game_state.agent_index < 0):
            correction = reward + self.discount_rate * 0 - self.get_qvalue(old_game_state, new_game_state, action)
        else:
            correction: float = ((reward + self.discount_rate * self.get_mdp_state_value(new_game_state))
                            - self.get_qvalue(old_game_state, new_game_state, action))
        # get all of the features
        features = self.feature_extractor(new_game_state) # unlike how pa3 does it, feature extractor wants the new state
        # print("updating state %s:" % (old_game_state.get_agent_position(0)))
        # print("Features: %s" % (features))
        # print("Current weights: %s" % (self.weights))
        # print("correction = %f = (%f + %f * %f) - %f" % (correction, reward, self.discount_rate,
        # self.get_mdp_state_value(old_game_state, old_game_state), self.get_qvalue(old_mdp_state, old_game_state, action)))
        for key, value in self.weights.items():
            # print("updating [%s: %f]" % (str(key), value))
            newWeight = value + self.learning_rate * correction * features.get(key, 0.0)
            # print("%f = %f + %f * %f * %f" % (newWeight, value, self.learning_rate, correction, features.get(key, 0.0)))
            self.weights[key] = newWeight

class Learning_Abel(capture.Abel):
    def __init__(self,
        **kwargs: typing.Any) -> None:
        # starting weights comes from parent class
        # feature extractor is also hardcoded to parent class
        super().__init__(**kwargs)
            
        #q learning related variables
        # these values are the defaults specified in pacai/agents/mdp.py
        # these aren't controlled by arguments
        self.discount_rate = 0.9
        self.learning_rate = 0.5
        self.exploration_rate = 0.3
        self.last_state: GameState | None = None
        self.total_rewards: float = 0.0
        self.training = True # we will always be training, but just in case I'm keeping the old infastructure



    def pack_training_info(self) -> dict[str, typing.Any]:
        return {
            'weights': self.weights,
    }

    def unpack_training_info(self, data: dict[str, typing.Any]) -> None:
        self.weights = Features.WeightDict(data.get('weights', {}))

    def game_complete(self, final_state: GameState) -> None:
        super().game_complete(final_state)
        logging.debug("Weights: %s.", edq.util.json.dumps(self.weights)) # very important for copying weights to capture.py

    def game_start(self, initial_state: GameState) -> None:
        self.last_state = initial_state

    # hopefully teh seperate game_complete and game_complete_full doesn't cause problems
    def game_complete_full(self,
            final_state: GameState,
            ) -> pacai.core.agentaction.AgentAction:
        if (self.training):
            logging.debug("Completed training epoch %d.", self.training_epoch)

        self.update(final_state)
        average_reward = 0.0
        num_actions = len(final_state.get_agent_actions(self.agent_index))
        if (num_actions > 0):
            average_reward = self.total_rewards / num_actions
        logging.debug("Made %d moves for a total of %0.2f rewards (average: %0.2f).",
                num_actions, self.total_rewards, average_reward)
        # Store the training information for the next epoch's agent.
        agent_action = super().game_complete_full(final_state)
        agent_action.training_info['training_info'] = self.pack_training_info()
        return agent_action

    def update(self, new_state: GameState) -> None:
        """
        Update the agent based on the difference between the old state and new state.
        """
        # Get the most recent action.
        # this is called before the new action is taken. So its the action that brought you to your current state
        last_action = new_state.get_last_agent_action(self.own_index) # new state won't be right after our agent's turn, but this
        # will still extract the right action
        if (last_action is None):
            # No action has been taken yet, don't update.
            return

        # Update the last seen state.
        old_state = self.last_state # this would be the last state where this agent took an action. new state is 3 turns later
        self.last_state = new_state.copy()

        if (old_state is None):
            # We don't have an old state to compare against yet.
            return

        # Compute and store the score delta.
        # remember to normalize it
        score_delta = new_state.get_normalized_score(self.own_index) - old_state.get_normalized_score(self.own_index)
        self.total_rewards += score_delta

        # Do not update if we are not training.
        if (not self.training):
            return

        old_position = old_state.get_agent_position(self.own_index)
        if (old_position is None):
            # The agent was not on the board the last turn. Did they respawn?
            return

        new_position = self.last_positions[-1]

        self.update_qvalue(score_delta, last_action,
            old_state, new_state,
            old_position, new_position)

    def get_action(self, state: GameState) -> action.Action:
        # Update the agent by learning from the environment.
        # This code should not change and always be the first thing done in this method.
        self.update(state)
        # use epilsion to calcuate if I do the policy action or a random one
        
        legal_actions: list[action.Action] = state.get_legal_actions()
        roll = self.rng.randrange(0, 100)
        if roll <= self.exploration_rate:
            # choose random
            a: action.Action = self.rng.choice(legal_actions)
            # print("Choosing random action " + str(a))
            return a
        else:
           # our policy is outlined in the parent classes, so just do super
           return super().get_action(state)

    def get_mdp_state_value(self, game_state: GameState) -> float:
        """Finds the maximum Q-value among all legal actions for a state."""
        # first check if terminal or agent doesn't exist. then I should just be able to return zero and it isn't even an error
        # print("Evaluating " + str(mdp_state))
        # looks toward the future more than the actual agent
        actions: list[action.Action] = game_state.get_legal_actions()
        print("in get_mdp_state_value:")
        print("legal actins: %s" % str(actions))
        maxValue = float('-inf')
        for a in actions:
            new_state = game_state.generate_successor(a)
            # have to make new state ourselves
            q = self.get_qvalue(game_state, new_state, a)
            print("%s: %f" % (str(a), q))
            if q > maxValue:
                maxValue = q
        return maxValue

    def get_qvalue(self, old_state, new_state,
        action: action.Action) -> float:
        # even when i'm not using old_state and action it just feels weird to call get_qvalue without them
        # just run evaluate
        q = self.evaluate(new_state)
        return q

    def update_qvalue(self,
        reward: float,
        action: action.Action,
        old_game_state: pacai.core.gamestate.GameState, new_game_state: pacai.core.gamestate.GameState,
        old_position: pacai.core.board.Position | None, new_position: pacai.core.board.Position | None,
        ) -> None:
        # since weights are being remembered instead of q values, update with:
        # wi ← wi + α ∗ [correction] ∗ fi(s,a)
        # correction = (R(s,a) + γ ∗ V'(s)) − Q(s,a)

        # correction = (R(s,a) + γ ∗ V'(s)) − Q(s,a)
        # since its s and not s', use the old state
        correction = 0.0
        if (new_game_state.agent_index < 0):
            correction = reward + self.discount_rate * 0 - self.get_qvalue(old_game_state, new_game_state, action)
        else:
            correction: float = ((reward + self.discount_rate * self.get_mdp_state_value(new_game_state))
                            - self.get_qvalue(old_game_state, new_game_state, action))
        # get all of the features
        features = self.feature_extractor(new_game_state) # unlike how pa3 does it, feature extractor wants the new state
        # print("updating state %s:" % (old_game_state.get_agent_position(0)))
        # print("Features: %s" % (features))
        # print("Current weights: %s" % (self.weights))
        print("correction = %f = (%f + %f * %f) - %f" % (correction, reward, self.discount_rate,
        self.get_mdp_state_value(new_game_state), self.get_qvalue(old_game_state, new_game_state, action)))
        for key, value in self.weights.items():
            # print("updating [%s: %f]" % (str(key), value))
            newWeight = value + self.learning_rate * correction * features.get(key, 0.0)
            print("%f = %f + %f * %f * %f" % (newWeight, value, self.learning_rate, correction, features.get(key, 0.0)))
            self.weights[key] = newWeight
        print("new weights are: %s" % str(self.weights))