from langgraph.graph import END, START, StateGraph

from admissions_agent.agent import nodes
from admissions_agent.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("regex_guard", nodes.regex_guard_node)
    graph.add_node("nlu_guard", nodes.nlu_guard_node)
    graph.add_node("rewrite_query", nodes.rewrite_query_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("build_context", nodes.build_context_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("output_guard", nodes.output_guard_node)
    graph.add_node("refuse", nodes.refuse_node)

    graph.add_edge(START, "regex_guard")
    graph.add_conditional_edges(
        "regex_guard", nodes.route_after_guard, {"continue": "nlu_guard", "refuse": "refuse"}
    )
    graph.add_conditional_edges(
        "nlu_guard", nodes.route_after_guard, {"continue": "rewrite_query", "refuse": "refuse"}
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "output_guard")
    graph.add_conditional_edges(
        "output_guard", nodes.route_after_guard, {"continue": END, "refuse": "refuse"}
    )
    graph.add_edge("refuse", END)

    return graph.compile()
