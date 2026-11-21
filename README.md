# Dual-engine push-optimal Sokoban solver (v2)

An entry for make.fun's "Build the best Sokoban solver, 21 November 2026"
bounty ($100 in USDC). This is a second, independently designed version of
the author's earlier room-by-room entry.

`main.ts` (Node 22, no dependencies) reads each level at
`/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`) and writes its moves to
`/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`).

How it works:

- The generated levels are a one-square-wide hallway with small rooms
  opening off it through single doors. A box pushed into the hallway can
  never reach a goal, so every box stays in its room and the level's push
  optimum is the sum of its rooms' optima. The solver cuts the hallway's
  dead squares out of the floor plan, splitting the board into one
  component per room, then solves each room and walks the player between
  rooms. A whole-board pass covers any level the split does not recognise.
- Every room is attacked by **two independent optimal engines**, and the
  better replay-verified result is kept:
  - **IDA\*** — iterative-deepening depth-first search over pushes, boxes
    tried in descending order and directions reversed, with a
    per-iteration transposition table.
  - **A\*** — best-first search over pushes ordered by pushes + estimate.
- A **greedy** best-first run goes first and supplies an **upper bound**:
  once a solution with U pushes is in hand, states whose pushes + estimate
  reach U are pruned, which makes the A* optimality proof far cheaper. If
  an optimal engine exhausts the space under that bound, the greedy
  solution is proven optimal; if both optimal engines run out of budget,
  the verified greedy solution still stands.
- Deadlock pruning goes beyond dead squares and frozen 2x2 blocks with
  **recursive freeze detection**: a box blocked on a horizontal side and a
  vertical side by walls or by boxes that are themselves frozen can never
  move again; the marking iterates to a fixed point, and a frozen box off
  its goal kills the state.
- The estimate is the cheapest assignment of boxes to distinct goals over
  lone-box push distances, computed by a **bitmask DP and cached per box
  tuple**, so repeat states reuse earlier assignment work and the
  expansion budgets stretch further.
- Every candidate move string is **replayed on the real board before it
  is used**; only a verified solution is ever written.
- All budgets count expanded states, never the clock, and every tie
  breaks by a fixed rule, so repeated runs write identical bytes.

On the kit's practice set it solves every level in exactly the optimal
number of pushes (13 / 16 / 29 / 47 / 60) for the maximum score, and on
randomly drawn sets it likewise matched the known optimum on every level
tested during development.
