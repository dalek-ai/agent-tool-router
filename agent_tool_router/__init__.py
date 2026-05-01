"""agent-tool-router — pick the right tools for an agent task.

>>> from agent_tool_router import Router
>>> r = Router.from_pretrained("baseline-v0")
>>> r.route("Find me cheap flights from Paris to NYC next month", k=3)
['search_flights', 'get_airport_code', 'compare_prices']
"""

from .router import Router

__version__ = "0.1.0"
__all__ = ["Router"]
