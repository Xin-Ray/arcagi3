<!-- COPIED from archive/outputs_2026-05-14/spatial_bench_v2/results.md; paths rewritten to point back to archive/outputs_2026-05-14/spatial_bench_v2/ -->
<!-- DO NOT EDIT; edit the original to refresh this copy. -->

# Spatial reasoning bench v2 — RAW vs ENRICHED prompt (text-only)

Model: Qwen2.5-VL-3B (4-bit), text-only mode.  Questions: 17.
Both configs share the same model and questions; only the prompt text differs.

## Summary by category

| Category | n | raw correct | enriched correct |
|---|---|---|---|
| **ACTION** | 3 | 1/3 | **2/3** |
| **COMPOUND** | 1 | 1/1 | **1/1** |
| **DIRECTION** | 5 | 0/5 | **5/5** |
| **POSITION** | 8 | 4/8 | **4/8** |

| **OVERALL** | 17 | 6/17 | **12/17** |

Wall-clock: raw 4.3s, enriched 4.0s.

## Per-question

### D1_up (DIRECTION)

**Q:** What direction did the object move? Answer one word only: up, down, left, or right.

**Expected:** `up`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `down` | ❌ | 0.61s |
| enriched | `up` | ✅ | 0.21s |

### D2_down (DIRECTION)

**Q:** What direction did the object move? Answer one word only.

**Expected:** `down`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `Right` | ❌ | 0.18s |
| enriched | `DOWN` | ✅ | 0.2s |

### D3_left (DIRECTION)

**Q:** What direction did the object move? Answer one word only.

**Expected:** `left`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `DOWN` | ❌ | 0.18s |
| enriched | `Left` | ✅ | 0.2s |

### D4_right (DIRECTION)

**Q:** What direction did the object move? Answer one word only.

**Expected:** `right`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `UP` | ❌ | 0.19s |
| enriched | `Right` | ✅ | 0.2s |

### D5_compound_up_left (DIRECTION)

**Q:** List all directions that describe this move (e.g. 'up and left'). Answer concisely.

**Expected:** `up + left`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `down and right` | ❌ | 0.34s |
| enriched | `UP, LEFT` | ✅ | 0.37s |

### A1_pick_up (ACTION)

**Q:** You want to move the yellow object further UP by another 3 cells. Which action token?

**Expected:** `ACTION1`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `ACTION2` | ❌ | 0.26s |
| enriched | `ACTION2` | ❌ | 0.3s |

### A2_pick_opposite (ACTION)

**Q:** Which action is likely OPPOSITE to ACTION3 (i.e. would move objects right)? Answer in the form ACTION_.

**Expected:** `ACTION4 (or other)`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `ACTION_LEFT` | ❌ | 0.26s |
| enriched | `ACTION_RIGHT` | ✅ | 0.28s |

### A3_dont_repeat_failed (ACTION)

**Q:** From legal actions {ACTION1, ACTION2, ACTION3, ACTION4}, which one should you try NEXT to gather new evidence? Answer one token.

**Expected:** `any of ACTION1/3/4`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `ACTION1` | ✅ | 0.26s |
| enriched | `ACTION1` | ✅ | 0.29s |

### P1_leftmost (POSITION)

**Q:** Among id=0 (yellow), id=1 (gray), id=5 (purple), which is the LEFTMOST? Answer with the color name only.

**Expected:** `gray`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `gray` | ✅ | 0.19s |
| enriched | `gray` | ✅ | 0.2s |

### P2_topmost (POSITION)

**Q:** Which color is the TOPMOST object (smallest row)? Answer one color name.

**Expected:** `yellow`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `red` | ❌ | 0.19s |
| enriched | `red` | ❌ | 0.18s |

### P3_largest (POSITION)

**Q:** Which id has the LARGEST size? Answer with the id number only.

**Expected:** `3`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `id=3` | ✅ | 0.35s |
| enriched | `3` | ❌ | 0.2s |

### P4_count_purple (POSITION)

**Q:** How many PURPLE objects are in the frame? Answer one integer.

**Expected:** `1`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `1` | ❌ | 0.18s |
| enriched | `1` | ❌ | 0.21s |

### P5_between (POSITION)

**Q:** Considering only id=0 (yellow), id=1 (gray), and id=5 (purple), is purple horizontally between gray and yellow? yes or no.

**Expected:** `yes`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `yes` | ✅ | 0.19s |
| enriched | `yes` | ✅ | 0.2s |

### P6_grid_quadrant_cd82 (POSITION)

**Q:** Which quadrant of the 64x64 grid is the GREEN object in? Choose one: top-left, top-right, bottom-left, bottom-right. (Quadrants split at row 32 and col 32.)

**Expected:** `bottom-right`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `bottom-left` | ❌ | 0.27s |
| enriched | `bottom-right` | ✅ | 0.26s |

### P7_count_objects_cn04 (POSITION)

**Q:** How many DISTINCT objects are in the frame? Answer one integer.

**Expected:** `4`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `4` | ❌ | 0.18s |
| enriched | `4` | ❌ | 0.18s |

### P8_adjacent_color (POSITION)

**Q:** Considering only id=0 (yellow), id=1 (gray), id=5 (purple): which has its center closest to row 19? Answer with the color name.

**Expected:** `yellow or gray (both at row 19)`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `gray` | ✅ | 0.18s |
| enriched | `gray` | ✅ | 0.2s |

### C1_push_yellow_to_wall (COMPOUND)

**Q:** Yellow id=0 is at center col 40. Purple wall id=5 is at center col 31. Which action token will push yellow TOWARD the purple wall?

**Expected:** `ACTION3`

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| raw      | `ACTION3` | ✅ | 0.26s |
| enriched | `ACTION3` | ✅ | 0.28s |
