---
date: 2025-10-04
course: test101
duration: 01:14:42
tags: [lecture, test101]
---

# Classroom Narrative

> [!note]
> This lesson focuses on implementing search algorithms in artificial intelligence. The instructor guides students through setting up a Wikipedia API wrapper as a tool and then transitions to a detailed explanation of search problems, agents, states, and the core concepts of Depth-First Search (DFS) and Breadth-First Search (BFS) algorithms.

## [00:00:00] Setting Up Wikipedia API as a Tool

The instructor begins by demonstrating how to configure a Wikipedia API wrapper to function as a tool within a larger system.

-   **`doc_content_chars_max`**: This parameter controls the maximum number of characters to retrieve from a Wikipedia page. For a quick demo, it's set to 100, but it can be increased to 1,000 or 10,000 for more content. Larger values may increase runtime and lead to faster rate limiting.
-   Other parameters like `language` and `load_all_available_metadata` can also be passed.

The API wrapper is then converted into a tool:
```python
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
```
This `wiki_tool` can be directly passed to LangChain without needing to be wrapped in a custom tool. The instructor then integrates this `wiki_tool` into a list of available tools.

To test, the system is prompted with "hammerhead sharks." The output shows that Wikipedia is used to look up "hammerhead shark" and then a search is performed for "hammerhead shark research latest findings," providing a response and indicating the use of two tools.

## [00:01:45] Creating a Custom Tool for Saving Data

The instructor then demonstrates how to create a custom tool to save data to a file. This involves writing a Python function that can be wrapped as a tool.

The function `save_to_TXT` is introduced:
```python
def save_to_TXT(data: str, file_name: str):
    # Function implementation to write data to a text file
    # Example: Writes research output, timestamp, and data
    pass
```
> [!important]
> It is crucial to provide type hints for parameters in custom tool functions so that the model understands how to call the function. In this example, `data` and `file_name` are typed as `str`.

The function is then wrapped as a tool:
```python
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_TXT,
    description="Save structured research data to a text file."
)
```
The `save_tool` is added to the list of available tools.

To test this custom tool, the system is prompted with "research South East Asia population and save to a file." The output indicates the use of Wikipedia and the `save_text_to_file` tool. A file named `research_output.txt` is created, containing a timestamp, the topic, and the research data, demonstrating the successful integration of the custom tool.

## [00:04:30] Introduction to Artificial Intelligence and Search Problems

The lesson transitions to a new topic: an introduction to artificial intelligence with Python, specifically focusing on search algorithms.

> [!important]
> **Artificial Intelligence (AI)**: Any instance where a computer performs a task that appears intelligent or rational, such as facial recognition, game playing, or natural language understanding.

The course will explore key AI concepts:
-   **Search**: Finding solutions to problems (e.g., driving directions, game moves).
-   **Knowledge**: Representing information and drawing inferences.
-   **Uncertainty**: Dealing with probabilistic information.
-   **Optimization**: Maximizing goals or minimizing costs.
-   **Machine Learning**: Learning from data and experience (e.g., spam detection).
-   **Neural Networks**: AI inspired by the human brain.
-   **Natural Language Processing (NLP)**: Understanding and interpreting human language.

The current focus is on **search problems**.

## [00:07:45] Defining Search Problems: Terminology

The instructor introduces fundamental terminology for understanding search problems.

> [!important]
> **Agent**: An entity that perceives its environment and acts upon it. (e.g., a car in driving directions, a person solving a puzzle).

> [!important]
> **State**: A specific configuration of the agent within its environment. (e.g., a particular arrangement of tiles in the 15-puzzle).

> [!important]
> **Initial State**: The starting configuration of the agent. This is the beginning point for any search algorithm.

> [!important]
> **Actions**: Choices that can be made in any given state. Formally defined as a function `actions(S)` that takes a state `S` and returns the set of all executable actions in that state.

> [!important]
> **Transition Model**: A description of the state resulting from performing an action in a given state. Formally defined as a function `result(S, A)` that takes a state `S` and an action `A`, and returns the new state after performing `A` in `S`.

The instructor illustrates the `result` function with the 15-puzzle, showing how an initial state and an action (e.g., sliding a tile right) produce a new state.

> [!important]
> **State Space**: The set of all possible states reachable from the initial state through any sequence of actions. This can be visualized as a graph where nodes are states and edges are actions.

> [!important]
> **Goal Test**: A mechanism to determine if a given state is a goal state. (e.g., reaching a destination, all numbers in order in the 15-puzzle).

> [!important]
> **Path Cost**: A numerical value assigned to a sequence of actions (a path) indicating its "expense" (e.g., time, distance, resources). The goal is often to find a solution that minimizes this cost.

> [!important]
> **Search Problem**: Defined by:
> -   An initial state.
> -   Actions available in any state.
> -   A transition model.
> -   A goal test.
> -   A path cost function.

> [!important]
> **Solution**: A sequence of actions that takes the agent from the initial state to a goal state.
> **Optimal Solution**: A solution with the lowest path cost among all possible solutions.

## [00:16:50] Nodes and the Search Algorithm Approach

To solve search problems, data needs to be organized.

> [!important]
> **Node**: A data structure used to keep track of:
> -   The current **state**.
> -   The **parent** node (the node from which the current state was reached).
> -   The **action** taken to reach the current state from the parent.
> -   The **path cost** from the initial state to the current state.

The core search algorithm approach:
1.  Initialize a **frontier** (a data structure containing states to be explored) with the initial state.
2.  Loop:
    a.  If the frontier is empty, there is no solution.
    b.  Remove a node from the frontier.
    c.  If the node is the goal state, a solution is found.
    d.  Otherwise, **expand** the node (find all reachable neighbor states) and add them to the frontier.
    e.  Repeat.

The instructor demonstrates this process with a simple graph (A to E).

## [00:20:30] Addressing Infinite Loops: The Explored Set

A potential problem with the basic search algorithm is getting stuck in infinite loops (e.g., going back and forth between two states).

To solve this, a new data structure is introduced:

> [!important]
> **Explored Set**: A set of nodes that have already been visited and expanded.

The revised search algorithm approach:
1.  Initialize a **frontier** with the initial state.
2.  Initialize an empty **explored set**.
3.  Loop:
    a.  If the frontier is empty, there is no solution.
    b.  Remove a node from the frontier.
    c.  If the node is the goal state, a solution is found.
    d.  Add the node to the **explored set**.
    e.  Expand the node and add resulting nodes to the frontier **only if they are not already in the frontier AND not in the explored set**.
    f.  Repeat.

## [00:22:45] Depth-First Search (DFS)

The choice of how to remove a node from the frontier is crucial.

> [!important]
> **Stack**: A Last-In, First-Out (LIFO) data structure. When the frontier is implemented as a stack, the most recently added node is explored first.

When the frontier is a stack, the algorithm performs a **Depth-First Search (DFS)**.

> [!important]
>