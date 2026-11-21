// Dual-engine push-optimal Sokoban solver (v2).
//
// Shape of the problem: the levels are a one-square-wide hallway with small
// rooms opening off it through single doors. A box pushed into the hallway
// can never be pushed onto a goal, so each box stays in its own room and a
// level's push optimum is the sum of its rooms' optima. This solver cuts the
// hallway's dead squares out of the floor plan to split the board into one
// component per room, then attacks every room with TWO independent optimal
// engines and keeps the better replay-verified result:
//
//   1. IDA* — an iterative-deepening depth-first search over pushes, in a
//      deliberately different traversal order (boxes descending, directions
//      reversed) with a per-iteration transposition table. With an admissible
//      estimate, the first solution it returns is push-optimal.
//   2. A* — a best-first search over pushes ordered by pushes + estimate,
//      pruned by any incumbent upper bound.
//
// A greedy best-first run goes first and supplies that upper bound: once a
// solution with U pushes is in hand, any state whose pushes + estimate reach
// U can be pruned, which makes the A* optimality proof much cheaper. If both
// optimal engines run out of budget, the greedy solution (always verified)
// is what stands. A whole-board pass with the same engines covers any level
// the room split does not recognise.
//
// Pruning, beyond dead squares (squares from which no lone box can reach a
// goal), uses RECURSIVE FREEZE detection: a box boxed in on one horizontal
// side and one vertical side by walls or by boxes that are themselves frozen
// can never move again; the marking is iterated to a fixed point, and a state
// where a frozen box rests off a goal is dead. Frozen 2x2 blocks are caught
// by the same test's companion scan.
//
// The estimate is the cheapest assignment of boxes to distinct goals over
// lone-box push distances, computed by a bitmask DP and cached per box tuple,
// so repeated states re-use earlier assignment work and the effective
// expansion budget goes further than a from-scratch estimate would allow.
//
// Every candidate string is replayed on the real board before it is used,
// and only a verified solution is ever written. All budgets count expanded
// states, never the clock, and every tie breaks by a fixed rule, so two runs
// write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Fixed direction table: [dx, dy, walk letter, push letter]. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];
/** The same directions in reverse — the IDA* engine's traversal order. */
const DIRS_REV: [number, number, string, string][] = [DIRS[3], DIRS[2], DIRS[1], DIRS[0]];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Sol {
  pushes: number;
  moves: string;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/** Lone-box push distances to each goal: a breadth-first pull-back per goal. */
function goalDistances(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  const size = wall.length;
  return goals.map((g) => {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const cur = queue[qi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const from = cur - d;
        const stand = from - d;
        if (from < 0 || from >= size || stand < 0 || stand >= size) continue;
        if (!wall[from] && !wall[stand] && dist[from] < 0) {
          dist[from] = dist[cur] + 1;
          queue.push(from);
        }
      }
    }
    return dist;
  });
}

/**
 * Admissible estimate: cheapest box-to-goal assignment over lone-box
 * distances, by bitmask DP (few boxes), cached by box tuple; for many boxes,
 * the sum of per-box minima. Infinity when some box reaches no goal.
 */
function makeEstimator(level: Level, dists: Int32Array[]) {
  const n = level.boxes.length;
  const m = level.goals.length;
  const useAssign = n <= 6 && n === m;
  const cache = new Map<string, number>();
  return (boxes: number[]): number => {
    const key = boxes.join(',');
    const hit = cache.get(key);
    if (hit !== undefined) return hit;
    let value: number;
    if (useAssign) {
      // dp(i, mask): cheapest cost of placing boxes i..n-1, goals in mask used.
      const memo = new Map<number, number>();
      const dp = (i: number, mask: number): number => {
        if (i === n) return 0;
        const mk = i * (1 << m) + mask;
        const seen = memo.get(mk);
        if (seen !== undefined) return seen;
        let best = Infinity;
        for (let j = 0; j < m; j++) {
          if (mask & (1 << j)) continue;
          const d = dists[j][boxes[i]];
          if (d < 0) continue;
          const rest = dp(i + 1, mask | (1 << j));
          if (rest < Infinity && d + rest < best) best = d + rest;
        }
        memo.set(mk, best);
        return best;
      };
      value = dp(0, 0);
    } else {
      let sum = 0;
      value = 0;
      for (const b of boxes) {
        let mn = Infinity;
        for (const dist of dists) {
          const d = dist[b];
          if (d >= 0 && d < mn) mn = d;
        }
        if (mn === Infinity) {
          value = Infinity;
          break;
        }
        sum += mn;
      }
      if (value !== Infinity) value = sum;
    }
    cache.set(key, value);
    return value;
  };
}

/** Player flood fill around walls and boxes, with walk reconstruction data. */
function flood(level: Level, occ: Uint8Array, from: number): { prev: Int32Array; how: string[]; rep: number } {
  const { w, wall } = level;
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length).fill('');
  prev[from] = from;
  let rep = from;
  const queue = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p < rep) rep = p;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length || wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  return { prev, how, rep };
}

function walkString(prev: Int32Array, how: string[], from: number, to: number): string {
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

function occupancy(size: number, boxes: number[]): Uint8Array {
  const occ = new Uint8Array(size);
  for (const b of boxes) occ[b] = 1;
  return occ;
}

/**
 * Deadlock test for a box configuration:
 *  - recursive freeze: iterate the marking "blocked on a horizontal side by a
 *    wall or an already-frozen box AND blocked on a vertical side likewise"
 *    to a fixed point; a frozen box off its goal can never move again.
 *  - any solid 2x2 of walls/boxes holding a box off its goal.
 */
function isDeadlocked(level: Level, occ: Uint8Array, goal: Uint8Array): boolean {
  const { w, wall } = level;
  const size = wall.length;
  const frozen = new Uint8Array(size);
  let changed = true;
  while (changed) {
    changed = false;
    for (let b = 0; b < size; b++) {
      if (!occ[b] || frozen[b]) continue;
      const horiz = wall[b - 1] || frozen[b - 1] || wall[b + 1] || frozen[b + 1];
      const vert = wall[b - w] || frozen[b - w] || wall[b + w] || frozen[b + w];
      if (horiz && vert) {
        frozen[b] = 1;
        changed = true;
      }
    }
  }
  for (let b = 0; b < size; b++) if (frozen[b] && occ[b] && !goal[b]) return true;
  for (let a = 0; a < size; a++) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (d >= size) continue;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

/** Replay a move string; null when any step is illegal. */
function replay(level: Level, moves: string): { player: number; boxes: number[]; pushes: number; solved: boolean } | null {
  const { w, wall } = level;
  const box = new Set(level.boxes);
  const goal = new Set(level.goals);
  let player = level.player;
  let pushes = 0;
  for (const ch of moves) {
    const dir = DIRS.find(([, , letter]) => letter === ch.toLowerCase());
    if (!dir) return null;
    const d = dir[1] * w + dir[0];
    const next = player + d;
    if (next < 0 || next >= wall.length || wall[next]) return null;
    if (box.has(next)) {
      const beyond = next + d;
      if (beyond < 0 || beyond >= wall.length || wall[beyond] || box.has(beyond)) return null;
      box.delete(next);
      box.add(beyond);
      pushes++;
    }
    player = next;
  }
  const boxes = [...box].sort((a, b) => a - b);
  return { player, boxes, pushes, solved: boxes.every((b) => goal.has(b)) };
}

function verified(level: Level, sol: Sol | null): Sol | null {
  if (!sol) return null;
  const check = replay(level, sol.moves);
  if (!check || !check.solved || check.pushes !== sol.pushes) return null;
  return sol;
}

interface BestFirstOutcome {
  sol: Sol | null;
  exhausted: boolean;
}

/**
 * Best-first search over pushes. Mode 'astar' orders by g + estimate and is
 * optimal; mode 'greedy' orders by estimate alone and returns the first
 * solution found. When `upperBound` is set, states that cannot beat it are
 * pruned; if the heap then empties, the bound is proven tight.
 */
function bestFirst(level: Level, cap: number, mode: 'astar' | 'greedy', upperBound: number): BestFirstOutcome {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dists = goalDistances(level);
  const live = new Uint8Array(size);
  for (const dist of dists) for (let i = 0; i < size; i++) if (dist[i] >= 0) live[i] = 1;
  const estimate = makeEstimator(level, dists);
  if (level.boxes.some((b) => !live[b])) return { sol: null, exhausted: true };
  if (level.boxes.every((b) => goal[b])) return { sol: { pushes: 0, moves: '' }, exhausted: true };
  const h0 = estimate(level.boxes);
  if (h0 === Infinity) return { sol: null, exhausted: true };

  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sG: number[] = [0];
  const sParent: number[] = [-1];
  const sMove: string[] = [''];
  const sH: number[] = [h0];
  const best = new Map<string, number>();
  {
    const occ = occupancy(size, level.boxes);
    const { rep } = flood(level, occ, level.player);
    best.set(String.fromCharCode(rep, ...level.boxes), 0);
  }
  const heap: number[] = [0];
  const rank = (s: number): number => (mode === 'astar' ? sG[s] + sH[s] : sH[s]);
  const less = (a: number, b: number): boolean => {
    const ra = rank(a);
    const rb = rank(b);
    if (ra !== rb) return ra < rb;
    if (sG[a] !== sG[b]) return sG[a] > sG[b];
    return a < b;
  };
  const heapPush = (s: number): void => {
    heap.push(s);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  const buildMoves = (s: number): string => {
    let moves = '';
    for (let t = s; t > 0; t = sParent[t]) moves = sMove[t] + moves;
    return moves;
  };

  let expanded = 0;
  while (heap.length > 0) {
    const s = heapPop();
    const boxes = sBoxes[s];
    if (boxes.every((b) => goal[b])) return { sol: { pushes: sG[s], moves: buildMoves(s) }, exhausted: false };
    if (++expanded > cap) return { sol: null, exhausted: false };
    const occ = occupancy(size, boxes);
    const { prev, how, rep } = flood(level, occ, sPlayer[s]);
    const known = best.get(String.fromCharCode(rep, ...boxes));
    if (known !== undefined && known < sG[s]) continue;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (const [dx, dy, , pushLetter] of DIRS) {
        const d = dy * w + dx;
        const stand = b - d;
        const dest = b + d;
        if (stand < 0 || stand >= size || dest < 0 || dest >= size) continue;
        if (prev[stand] === -1 || wall[dest] || occ[dest] || !live[dest]) continue;
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((x, y) => x - y);
        const nocc = occupancy(size, next);
        if (isDeadlocked(level, nocc, goal)) continue;
        const h = estimate(next);
        if (h === Infinity) continue;
        const g = sG[s] + 1;
        if (mode === 'astar' && g + h >= upperBound) continue;
        const sub = flood(level, nocc, b);
        const k = String.fromCharCode(sub.rep, ...next);
        const seenG = best.get(k);
        if (seenG !== undefined && seenG <= g) continue;
        best.set(k, g);
        const t = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sG.push(g);
        sParent.push(s);
        sMove.push(walkString(prev, how, sPlayer[s], stand) + pushLetter);
        sH.push(h);
        heapPush(t);
        if (mode === 'greedy' && next.every((bb) => goal[bb])) {
          return { sol: { pushes: g, moves: buildMoves(t) }, exhausted: false };
        }
      }
    }
  }
  return { sol: null, exhausted: true };
}

/**
 * The second, independent optimal engine: IDA* over pushes. Depth-first with
 * an f-bound that rises to the smallest exceeded value each iteration, a
 * transposition table of the cheapest visit per state per iteration, boxes
 * tried in descending order and directions reversed relative to A*.
 */
function idaStar(level: Level, cap: number, upperBound: number): Sol | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dists = goalDistances(level);
  const live = new Uint8Array(size);
  for (const dist of dists) for (let i = 0; i < size; i++) if (dist[i] >= 0) live[i] = 1;
  const estimate = makeEstimator(level, dists);
  if (level.boxes.some((b) => !live[b])) return null;
  if (level.boxes.every((b) => goal[b])) return { pushes: 0, moves: '' };
  const h0 = estimate(level.boxes);
  if (h0 === Infinity) return null;

  let expanded = 0;
  let found: Sol | null = null;

  const search = (boxes: number[], player: number, g: number, bound: number, path: string, seen: Map<string, number>): number => {
    if (found) return -1;
    if (++expanded > cap) return Infinity;
    const occ = occupancy(size, boxes);
    const { prev, how, rep } = flood(level, occ, player);
    const key = String.fromCharCode(rep, ...boxes);
    const seenG = seen.get(key);
    if (seenG !== undefined && seenG <= g) return Infinity;
    seen.set(key, g);
    let nextBound = Infinity;
    // Boxes descending, directions reversed: a genuinely different order.
    for (let bi = boxes.length - 1; bi >= 0; bi--) {
      const b = boxes[bi];
      for (const [dx, dy, , pushLetter] of DIRS_REV) {
        const d = dy * w + dx;
        const stand = b - d;
        const dest = b + d;
        if (stand < 0 || stand >= size || dest < 0 || dest >= size) continue;
        if (prev[stand] === -1 || wall[dest] || occ[dest] || !live[dest]) continue;
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((x, y) => x - y);
        const nocc = occupancy(size, next);
        if (isDeadlocked(level, nocc, goal)) continue;
        const h = estimate(next);
        if (h === Infinity) continue;
        const f = g + 1 + h;
        if (f > bound) {
          if (f < nextBound) nextBound = f;
          continue;
        }
        if (f >= upperBound) continue;
        const seg = walkString(prev, how, player, stand) + pushLetter;
        if (next.every((bb) => goal[bb])) {
          found = { pushes: g + 1, moves: path + seg };
          return -1;
        }
        const r = search(next, b, g + 1, bound, path + seg, seen);
        if (found) return -1;
        if (r < nextBound) nextBound = r;
        if (expanded > cap) return Infinity;
      }
    }
    return nextBound;
  };

  let bound = h0;
  while (!found && bound < upperBound && expanded <= cap) {
    const seen = new Map<string, number>();
    const r = search(level.boxes.slice(), level.player, 0, bound, '', seen);
    if (found) break;
    if (r === Infinity || r <= bound) break;
    bound = r;
  }
  return found;
}

/**
 * Solve one (sub)level with the full pipeline: greedy upper bound first,
 * then IDA*, then A* with the bound; keep the best verified result.
 */
function solveBoard(level: Level, idaCap: number, astarCap: number): Sol | null {
  const goal = new Uint8Array(level.wall.length);
  for (const g of level.goals) goal[g] = 1;
  if (level.boxes.every((b) => goal[b])) return { pushes: 0, moves: '' };

  let incumbent: Sol | null = verified(level, bestFirst(level, 300_000, 'greedy', Infinity).sol);

  if (level.boxes.length <= 5) {
    const ida = verified(level, idaStar(level, idaCap, incumbent ? incumbent.pushes : Infinity));
    if (ida) return ida; // IDA* with an admissible estimate is optimal.
  }
  const ub = incumbent ? incumbent.pushes : Infinity;
  const a = bestFirst(level, astarCap, 'astar', ub);
  const aSol = verified(level, a.sol);
  if (aSol) return aSol; // A* with an admissible estimate is optimal.
  if (a.exhausted) return incumbent; // No better solution exists: the bound was tight.
  // Neither optimal engine finished: fall back to a bigger greedy effort.
  if (!incumbent) incumbent = verified(level, bestFirst(level, 900_000, 'greedy', Infinity).sol);
  return incumbent;
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/** Split into rooms by cutting the dead squares of the player's hallway row. */
function splitRooms(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const dists = goalDistances(level);
  const live = new Uint8Array(wall.length);
  for (const dist of dists) for (let i = 0; i < wall.length; i++) if (dist[i] >= 0) live[i] = 1;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const comps: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const id = comps.length;
    const squares = [s];
    part[s] = id;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (q < 0 || q >= wall.length || wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = id;
        squares.push(q);
      }
    }
    comps.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) comps[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    comps[part[g]].goals.push(g);
  }
  const rooms: Room[] = [];
  for (const c of comps) {
    if (c.boxes.length === 0) continue;
    if (c.boxes.length !== c.goals.length) return null;
    rooms.push({ boxes: c.boxes.sort((a, b) => a - b), goals: c.goals.sort((a, b) => a - b), done: false });
  }
  return rooms.length > 0 ? rooms : null;
}

function currentBoxes(level: Level, rooms: Room[]): number[] {
  const out = new Set(level.boxes);
  for (const r of rooms) {
    if (!r.done) continue;
    for (const b of r.boxes) out.delete(b);
    for (const g of r.goals) out.add(g);
  }
  return [...out].sort((a, b) => a - b);
}

/** Room-by-room solve; other rooms' boxes (or finished goals) stand as walls. */
function solveByRooms(level: Level): Sol | null {
  const rooms = splitRooms(level);
  if (!rooms) return null;
  let player = level.player;
  let moves = '';
  let pushes = 0;
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const cell of other.done ? other.goals : other.boxes) wall[cell] = 1;
    }
    const sub: Level = { w: level.w, h: level.h, wall, goals: room.goals, boxes: room.boxes, player };
    const solved = solveBoard(sub, 250_000, 2_500_000);
    if (!solved) return null;
    const state = replay({ ...level, boxes: currentBoxes(level, rooms), player }, solved.moves);
    if (!state) return null;
    player = state.player;
    moves += solved.moves;
    pushes += solved.pushes;
    room.done = true;
  }
  const check = replay(level, moves);
  if (!check || !check.solved || check.pushes !== pushes) return null;
  return { pushes, moves };
}

function solve(text: string): string {
  const level = parseLevel(text);
  const byRooms = solveByRooms(level);
  if (byRooms) return byRooms.moves;
  const whole = solveBoard(level, 250_000, 1_500_000);
  if (whole) {
    const check = replay(level, whole.moves);
    if (check && check.solved) return whole.moves;
  }
  return '';
}

for (const id of readdirSync(IN_DIR).sort()) {
  const moves = solve(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
