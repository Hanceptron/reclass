---
date: 2025-10-04
course: test101
duration: 01:14:42
tags: [lecture, test101]
---

# AI and Search Algorithms — Emirhan's Class Notes

## Personal Overview
Today’s session framed Artificial Intelligence as systematic problem-solving and drilled into **state-space search**. We compared how **DFS** and **BFS** behave, when they succeed, and the bookkeeping (nodes, frontier, explored set) that keeps them efficient. Treat this lecture as the backbone for the first midterm—later topics will layer heuristics, so mastering these ideas now saves stress.

## Lecture Timeline & Key Points
1. **00:00–04:20 — What counts as AI?** Examples (spam filters, self-driving cars, translation) anchor the definition. Expect a short-answer exam question here.
2. **04:20–15:20 — Anatomy of a Search Problem.** Agent, state, actions, transition model, goal test, path cost. You should be able to restate these in your own words and give a fresh example (e.g., robot vacuum cleaning an apartment). 
3. **15:20–22:00 — Nodes, Frontier, Explored Set.** Understand the node structure (state, parent, action, path cost). Visualize the frontier as the "to-do" list; explored set prevents loops.
4. **22:00–30:00 — DFS Mechanics.** Stack-based, can dive deep fast but risks missing the optimal solution or getting stuck in infinite depth without safeguards.
5. **30:00–40:00 — BFS Mechanics.** Queue-based, guarantees shortest path when costs are uniform but uses more memory. Instructor emphasized drawing tree levels for clarity.
6. **40:00–50:00 — Comparing DFS vs BFS.** Trade-offs in completeness, optimality, time/space. Think about test scenarios where you must recommend one over the other.
7. **50:00–End — Case Studies (15-puzzle, maze, driving directions).** Translate abstract definitions into concrete problems; practice drawing state graphs.

## Concepts to Master (with Triggers)
> [!important] Search Problem Template [04:50]
> Initial state → actions → transition model → goal test → path cost → solution
>
> **Homework idea:** Fill the template for a familiar task (e.g., planning your commute).

> [!note] Node Anatomy [16:00]
> `Node = (state, parent, action, path_cost)` — trace it on paper when simulating algorithms.

> [!warning] DFS Pitfall [27:30]
> Without an explored set or depth limit, DFS can revisit states forever. Use this in quiz answers when asked about limitations.

> [!tip] BFS Strength [32:10]
> First optimal solution when costs are uniform—mention this whenever asked “why BFS?”

## Midterm & Project Watchlist
- **Midterm 1:** Definitions (agent/state/goal/path cost), compare DFS vs BFS, simulate one iteration of each on a small graph.
- **Coding Assignment (likely next week):** Implement BFS first (because of optimality), then extend to DFS. Instructor hinted at a maze or sliding puzzle solver.
- **Project Brainstorm:** Keep notes on real problems where search applies; you may need to propose one for the term project.

## Pay Extra Attention / Gotchas
- Remember **frontier implementations** (stack vs queue) and how they alter behavior.
- **Space complexity** talk: BFS can exhaust memory quickly—use this when asked to critique algorithms.
- **Path cost vs depth:** DFS ignores cost; BFS assumes uniform cost. If costs vary, neither guarantees optimal solutions—flag this nuance.

## Study Checklist
- [ ] Re-write the search problem components with your own driving directions example.
- [ ] Hand-execute DFS and BFS on a 5-node graph; record the order of state expansion.
- [ ] Implement the 15-puzzle neighbor generator (pseudo-code acceptable).
- [ ] Summarize the differences between DFS/BFS (table format) and memorize.
- [ ] Prepare two questions to ask next class (e.g., “How does iterative deepening fix DFS limitations?”).

## Exam & Quiz Reminders
- Expect a **quiz next week**: likely to ask for frontier states after two steps of DFS/BFS.
- **Midterm** will include: evaluating algorithm properties, identifying failures (infinite loops), and describing fixes (depth limits, explored sets).
- Potential **extra credit**: applying search to an everyday task (instructor hinted at this).

## Homework / Practice Targets
- Code both DFS and BFS; print expanded nodes to verify understanding.
- Build a comparison table:
  | Algorithm | Data Structure | Completeness | Optimality | Time | Space |
  |-----------|----------------|--------------|------------|------|-------|
  | DFS       | Stack (LIFO)    | Depends on finiteness | Not optimal | O(b^m) | O(m) |
  | BFS       | Queue (FIFO)    | Always        | Optimal (uniform cost) | O(b^(d+1)) | O(b^(d+1)) |
- Challenge yourself: modify BFS to track path cost even when costs differ.

## Prep for Next Class
- Review heuristic search preview in the textbook—knowing this foundation will make A* less intimidating.
- Read CS50 AI notes on “Search” (sections on state-space representation).
- Bring laptop with Python ready; we may live-code BFS.

## Quick Reference
- **Frontier Choice:** Stack = DFS (deep first), Queue = BFS (level order).
- **Explored Set:** always use when the graph is cyclic.
- **Optimality Reminder:** BFS optimal only when all actions cost the same.
- **Practical Cue:** If memory is tight → DFS; if you need the shortest path → BFS.

Keep this page open in Obsidian during class so you can tick off the checklist and add timestamps/questions in real time.
