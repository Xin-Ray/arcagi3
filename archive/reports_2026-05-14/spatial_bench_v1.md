<!-- COPIED from archive/outputs_2026-05-14/spatial_bench/results.md; paths rewritten to point back to archive/outputs_2026-05-14/spatial_bench/ -->
<!-- DO NOT EDIT; edit the original to refresh this copy. -->

# Spatial reasoning benchmark — with image vs text only

Model: Qwen2.5-VL-3B (4-bit). Test image: `outputs\scipy_object_diag\ar25-0c556536\frame_01.png`.
Both modes share the exact same system prompt and user prompt — only the image content block is dropped in text-only mode.

## Summary

| Mode | Correct | Total seconds | Avg s/question |
|---|---|---|---|
| with_image | **4/6** | 3.0s | 0.51s |
| text_only  | **4/6** | 1.5s | 0.26s |

## Per-question

### Q1_leftright

**Q:** Between yellow (id=0) and gray (id=1), which one is to the LEFT? Answer with only the color name.

**Expected:** `gray`  (gray center col 22 < yellow center col 40)

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| with_image | `gray` | ✅ | 0.89s |
| text_only  | `gray` | ✅ | 0.19s |

### Q2_size_compare

**Q:** Which object has the largest size (most cells)? Answer with only the id number.

**Expected:** `3`  (maroon id=3 has size=1845, largest)

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| with_image | `id=7` | ❌ | 0.51s |
| text_only  | `id=3` | ✅ | 0.35s |

### Q3_between

**Q:** Is the purple object (id=5) horizontally BETWEEN gray id=1 and yellow id=0? Answer only yes or no.

**Expected:** `yes`  (gray col 22 < purple col 31 < yellow col 40)

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| with_image | `Yes` | ✅ | 0.34s |
| text_only  | `Yes` | ✅ | 0.18s |

### Q4_motion_direction

**Q:** Yellow id=0 had center (22.0, 40.0) in the previous frame and is now at center (19.0, 40.0). What direction did it move? Answer with only one word: up, down, left, or right.

**Expected:** `up`  (row decreased from 22 to 19, that's up (lower row index = up in image))

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| with_image | `left` | ❌ | 0.35s |
| text_only  | `left` | ❌ | 0.2s |

### Q5_action_pick

**Q:** From the previous step we observed: ACTION1 moved yellow id=0 from row 22 to row 19 (3 cells up). You now want to push yellow further up by 3 more cells. Which ACTION should you take? Answer with the action token only, e.g. ACTION1.

**Expected:** `ACTION1`  (evidence shows ACTION1 -> up movement)

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| with_image | `ACTION1` | ✅ | 0.46s |
| text_only  | `ACTION1` | ✅ | 0.27s |

### Q6_adjacency

**Q:** Which object's right edge is exactly adjacent to (one column from) the purple object's left edge? Answer with only the id number.

**Expected:** `1`  (gray id=1 col_max=26, purple col_min=30; gap of 3. Actually closest left neighbour is gray id=1 (col 26) — 3 cols gap. (purple right=32, yellow left=36 — gap of 3.) Both equidistant. Trick: gray id=1 right=26 vs purple left=30 = 3-col gap, no direct adjacency. Accept either id 1 or id 0 — they're both 3 cols away. Mark correct if the model picks id=1 (the literal one nearest by col_max).)

| Mode | Answer | Correct | Latency |
|---|---|---|---|
| with_image | `id=1` | ✅ | 0.49s |
| text_only  | `id=6` | ❌ | 0.35s |
