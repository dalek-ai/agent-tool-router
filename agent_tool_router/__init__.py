"""agent-tool-router — pick the right tools for an agent task.

>>> from agent_tool_router import Router
>>> r = Router.from_pretrained("baseline-v1-desc")
>>> r.route("cancel my pending order and refund the credit", k=3)
['refundOrder', 'modify_pending_order_items', 'cancel_pending_order']
"""

from .router import Router

__version__ = "0.1.0"
__all__ = ["Router"]
