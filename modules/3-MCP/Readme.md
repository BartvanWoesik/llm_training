# MCP Training Module

## Overview
In this module, we explore the Modular Component Protocol (MCP), a powerful architecture pattern for building scalable, composable, and interoperable AI agents and services. MCP enables the creation of modular services that communicate via standardized interfaces, making it easier to manage complexity and foster reuse across different domains.
What You'll Build
We set up two MCP servers:

Review Server: Handles review-related tasks.
Reservation Server: Manages reservation workflows.

As part of the assignment, you'll extend this setup by creating a third MCP server dedicated to creating and collecting movie reviews.

## Benefits of MCP

**Modularity**: Each server encapsulates a specific domain, making it easier to maintain and evolve independently.

**Interoperability**: MCP servers communicate via standardized interfaces, allowing seamless integration.

**Scalability**: New capabilities can be added by spinning up new MCP servers without disrupting existing ones.

**Reusability**: Components can be reused across different agents or workflows.



## create_react_agent from LangChain
We use LangChain’s [create_react_agent](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/) to build agents that can reason and act using tools exposed by MCP servers.
What it does:

Wraps tools (e.g., MCP endpoints) into a LangChain agent.
Enables agents to think step-by-step and call tools as needed.
Supports ReAct (Reasoning + Acting) paradigm for more intelligent decision-making.

Example usage:
```python
from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools, verbose=True)
response = agent.run("Can you reserve a table and leave a review?")
```

## Assignment
Your task is to:

Create a new MCP server that allows users to:

Submit reviews for movies.
Retrieve existing reviews.


Suggested Steps

Define the server schema (e.g., /submit_review, /get_reviews).
Implement the endpoints using FastAPI or another framework.
Register the server with your agent setup.
Test the integration by querying the agent.

## Extra's

Process the submitted review with the review script!