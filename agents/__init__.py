# from agents.fql import ACFQLAgent
from agents.fql import ACFQLAgent
from agents.fql_ptr import ACFQLAgent_PTR
from agents.sac import SACAgent

agents = dict(
    acfql=ACFQLAgent,
    acfql_ptr=ACFQLAgent_PTR,
    sac=SACAgent,
)
