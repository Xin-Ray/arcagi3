# Trace balance analysis

- trace files scanned: **73**
- total step rows: **6132**
- overall change_rate: **42.4%** (2599 changed / 3533 no-op)

## Per-run change_rate

| run | rows | change_rate | balance |
|---|---:|---:|---|
| ablation_overnight_ | 2400 | 69.3% | balanced |
| preload_v3_budget_smoke_20260515-004233 | 1000 | 3.2% | no-op heavy |
| bug8_9_smoke_20260514-205044 | 732 | 11.7% | no-op heavy |
| v3_p0b_p1_full | 400 | 53.5% | balanced |
| all_fixes_smoke_20260514-224124 | 319 | 38.6% | balanced |
| postrefactor_smoke_20260514-235011 | 239 | 2.5% | no-op heavy |
| v3_2_ar25_5x80 | 175 | 38.3% | balanced |
| v3_2_ar25_2x80_ABCD | 132 | 75.0% | change heavy |
| preload_map_smoke_20260515-002417 | 123 | 8.9% | no-op heavy |
| bug11_12_13_smoke_20260515-000528 | 106 | 10.4% | no-op heavy |
| v3_2_ar25_2x80 | 91 | 19.8% | no-op heavy |
| v3_2_ar25_3x30 | 90 | 18.9% | no-op heavy |
| v3_2_ar25_3x30_v2 | 90 | 85.6% | change heavy |
| v3_2_ar25_3x30_v3 | 90 | 72.2% | change heavy |
| v3_2_ar25_smoke_ABCD | 30 | 46.7% | balanced |
| preload_map_v2_smoke_20260515-003754 | 25 | 76.0% | change heavy |
| v3_2_ar25_smoke_R4 | 10 | 80.0% | change heavy |
| v3_2_ar25_smoke_R4b | 10 | 80.0% | change heavy |
| v3_2_ar25_smoke_R67_rel | 10 | 80.0% | change heavy |
| v3_2_smoke | 10 | 100.0% | change heavy |
| v3_2_smoke_real | 10 | 50.0% | balanced |
| mask_revive_3x200_20260516-014356 | 9 | 66.7% | balanced |
| mask_smoke_dry_20260516-014252 | 8 | 100.0% | change heavy |
| v3_2_dryrun_R2 | 8 | 100.0% | change heavy |
| v3_2_dryrun_ABCD | 5 | 100.0% | change heavy |
| v3_2_dryrun_ABCD2 | 5 | 100.0% | change heavy |
| v3_2_dryrun_R67_rel | 5 | 100.0% | change heavy |

## Per-(game, action) breakdown

| game | action | n_tried | n_changed | change_rate |
|---|---|---:|---:|---:|
| ar25-0c556536 | ACTION1 | 260 | 35 | 13.5% |
| ar25-0c556536 | ACTION2 | 10 | 10 | 100.0% |
| ar25-0c556536 | ACTION3 | 13 | 13 | 100.0% |
| ar25-0c556536 | ACTION4 | 40 | 25 | 62.5% |
| ar25-0c556536 | ACTION5 | 151 | 151 | 100.0% |
| ar25-0c556536 | ACTION6 | 66 | 0 | 0.0% |
| ar25-0c556536 | ACTION7 | 18 | 17 | 94.4% |
| ar25-0c556536 | RESET | 2 | 2 | 100.0% |
| bp35-0a0ad940 | ACTION1 | 17 | 17 | 100.0% |
| bp35-0a0ad940 | ACTION2 | 12 | 12 | 100.0% |
| bp35-0a0ad940 | ACTION3 | 41 | 41 | 100.0% |
| bp35-0a0ad940 | ACTION4 | 267 | 267 | 100.0% |
| bp35-0a0ad940 | ACTION5 | 10 | 6 | 60.0% |
| bp35-0a0ad940 | ACTION6 | 154 | 44 | 28.6% |
| bp35-0a0ad940 | ACTION7 | 46 | 17 | 37.0% |
| bp35-0a0ad940 | RESET | 13 | 13 | 100.0% |
| bug8_9_smoke_20260514-205044 | ACTION1 | 295 | 38 | 12.9% |
| bug8_9_smoke_20260514-205044 | ACTION2 | 28 | 28 | 100.0% |
| bug8_9_smoke_20260514-205044 | ACTION3 | 68 | 12 | 17.6% |
| bug8_9_smoke_20260514-205044 | ACTION4 | 2 | 2 | 100.0% |
| bug8_9_smoke_20260514-205044 | ACTION5 | 2 | 2 | 100.0% |
| bug8_9_smoke_20260514-205044 | ACTION6 | 333 | 0 | 0.0% |
| bug8_9_smoke_20260514-205044 | ACTION7 | 4 | 4 | 100.0% |
| cd82-fb555c5d | ACTION1 | 91 | 61 | 67.0% |
| cd82-fb555c5d | ACTION2 | 10 | 9 | 90.0% |
| cd82-fb555c5d | ACTION3 | 10 | 8 | 80.0% |
| cd82-fb555c5d | ACTION4 | 162 | 107 | 66.0% |
| cd82-fb555c5d | ACTION5 | 182 | 117 | 64.3% |
| cd82-fb555c5d | ACTION6 | 96 | 60 | 62.5% |
| cd82-fb555c5d | ACTION7 | 9 | 7 | 77.8% |
| cn04-2fe56bfb | ACTION1 | 35 | 31 | 88.6% |
| cn04-2fe56bfb | ACTION2 | 19 | 19 | 100.0% |
| cn04-2fe56bfb | ACTION3 | 16 | 12 | 75.0% |
| cn04-2fe56bfb | ACTION4 | 13 | 13 | 100.0% |
| cn04-2fe56bfb | ACTION5 | 320 | 316 | 98.8% |
| cn04-2fe56bfb | ACTION6 | 138 | 94 | 68.1% |
| cn04-2fe56bfb | ACTION7 | 12 | 5 | 41.7% |
| cn04-2fe56bfb | RESET | 7 | 7 | 100.0% |
| dc22-fdcac232 | ACTION1 | 177 | 101 | 57.1% |
| dc22-fdcac232 | ACTION2 | 32 | 31 | 96.9% |
| dc22-fdcac232 | ACTION3 | 32 | 28 | 87.5% |
| dc22-fdcac232 | ACTION4 | 197 | 112 | 56.9% |
| dc22-fdcac232 | ACTION5 | 9 | 6 | 66.7% |
| dc22-fdcac232 | ACTION6 | 101 | 57 | 56.4% |
| dc22-fdcac232 | ACTION7 | 12 | 7 | 58.3% |
| round_00 | ACTION1 | 847 | 171 | 20.2% |
| round_00 | ACTION2 | 64 | 64 | 100.0% |
| round_00 | ACTION3 | 85 | 37 | 43.5% |
| round_00 | ACTION4 | 18 | 18 | 100.0% |
| round_00 | ACTION5 | 9 | 9 | 100.0% |
| round_00 | ACTION6 | 466 | 0 | 0.0% |
| round_00 | ACTION7 | 268 | 115 | 42.9% |
| round_01 | ACTION1 | 327 | 62 | 19.0% |
| round_01 | ACTION2 | 23 | 23 | 100.0% |
| round_01 | ACTION3 | 260 | 18 | 6.9% |
| round_01 | ACTION4 | 18 | 18 | 100.0% |
| round_01 | ACTION5 | 14 | 14 | 100.0% |
| round_01 | ACTION6 | 79 | 0 | 0.0% |
| round_01 | ACTION7 | 17 | 17 | 100.0% |
| round_02 | ACTION1 | 21 | 13 | 61.9% |
| round_02 | ACTION2 | 7 | 7 | 100.0% |
| round_02 | ACTION3 | 18 | 13 | 72.2% |
| round_02 | ACTION4 | 25 | 25 | 100.0% |
| round_02 | ACTION5 | 8 | 8 | 100.0% |
| round_02 | ACTION6 | 23 | 0 | 0.0% |
| round_02 | ACTION7 | 3 | 3 | 100.0% |

## Balanced sample availability per action

Per action: min(n_changed, n_no_op) is the max class-balanced sample count.

| action | total n | n_changed | n_no_op | max balanced |
|---|---:|---:|---:|---:|
| ACTION1 | 2070 | 529 | 1541 | 529 |
| ACTION2 | 205 | 203 | 2 | 2 |
| ACTION3 | 543 | 182 | 361 | 182 |
| ACTION4 | 742 | 587 | 155 | 155 |
| ACTION5 | 705 | 629 | 76 | 76 |
| ACTION6 | 1456 | 255 | 1201 | 255 |
| ACTION7 | 389 | 192 | 197 | 192 |
| RESET | 22 | 22 | 0 | 0 |
