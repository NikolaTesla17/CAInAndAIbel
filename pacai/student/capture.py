import pacai.core.agentinfo
import pacai.capture.team
import pacai.util.alias

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    # return pacai.capture.team.create_team_dummy()
    
    agent1_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)
    agent2_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)

    return [agent1_info, agent2_info]
