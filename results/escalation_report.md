# Three-priority escalation policy — DT-rule Priority 1, CSV pool evaluation

**Honesty summary, stated plainly per the task's requirement**: Priority 1 is supervised signature matching (tree.txt's own attack leaves compiled into rules) with NO zero-day property. Priority 2's benign-only signature table is the ONLY component here carrying a zero-day claim (asserted at construction, same as `dataplane/fitting.py`). Priority 3 guarantees nothing — it improves the odds of observing novel behaviour, it does not detect it.

## Feature-set and class-coverage discrepancies found before this evaluation ran

- tree.txt's actual 10 features differ from the task brief's stated list in 3 of 10 (see `eval/parse_tree.py`); resolved with the user ("use tree.txt as ground truth") before any rule compilation.
- tree.txt covers **12 classes** (BENIGN + 11 attack classes), not 8. **Bot and Infiltration DO have tree leaves** (2 and 3 respectively) — contrary to the brief's framing, they are not unseen-by-construction for Priority 1. Only Heartbleed (0 leaves) and the 3-way Web Attack sub-label distinction (tree predicts one undifferentiated `Web Attack` class) are genuinely unseen by Priority 1's own leaf vocabulary. The pool has 15 ground-truth labels total, matching the brief.

## Priority 1 alone (no meter, no P2/P3) — from `results/p1_rules.md`

- Escalation rate: 19.8269% (561,249/2,830,743)
- Pooled attack recall: 87.8808%
- Benign FPR: 3.1316%
- This is far above any realistic 1% budget on its own — the meter and priority ordering below are what make the policy budget-respecting, not the rule set itself.

## Priority 2 — signature table fit and bin-resolution sweep

Fit population: Monday+Tuesday chronological FIRST HALF, BENIGN rows only (n=482,016; asserted benign-only at `build_signature_table`'s first check). Held-out same-period population (never touched by fitting): Monday+Tuesday chronological second half (n=487,914).

### Bin-resolution sweep — does PortScan recover with finer top bins?

|   n_bins |   n_common_signatures |   Bot__n |   Bot__recall |   DDoS__n |   DDoS__recall |   DoS GoldenEye__n |   DoS GoldenEye__recall |   DoS Hulk__n |   DoS Hulk__recall |   DoS Slowhttptest__n |   DoS Slowhttptest__recall |   DoS slowloris__n |   DoS slowloris__recall |   FTP-Patator__n |   FTP-Patator__recall |   Heartbleed__n |   Heartbleed__recall |   Infiltration__n |   Infiltration__recall |   PortScan__n |   PortScan__recall |   SSH-Patator__n |   SSH-Patator__recall |   Web Attack – Brute Force__n |   Web Attack – Brute Force__recall |   Web Attack – Sql Injection__n |   Web Attack – Sql Injection__recall |   Web Attack – XSS__n |   Web Attack – XSS__recall |
|---------:|----------------------:|---------:|--------------:|----------:|---------------:|-------------------:|------------------------:|--------------:|-------------------:|----------------------:|---------------------------:|-------------------:|------------------------:|-----------------:|----------------------:|----------------:|---------------------:|------------------:|-----------------------:|--------------:|-------------------:|-----------------:|----------------------:|------------------------------:|-----------------------------------:|--------------------------------:|-------------------------------------:|----------------------:|---------------------------:|
|        6 |                   403 |     1966 |      0.641404 |    128027 |    2.34326e-05 |              10293 |               0.0345866 |        231073 |         0          |                  5499 |                   0.30351  |               5796 |                0.385266 |             7938 |              0        |              11 |                    0 |                36 |               0.861111 |        158930 |        0.000232807 |             5897 |              0        |                          1507 |                           1        |                              21 |                             1        |                   652 |                  1         |
|       10 |                   985 |     1966 |      0.228891 |    128027 |    1           |              10293 |               0.0347809 |        231073 |         9.5208e-05 |                  5499 |                   0.336425 |               5796 |                0.385611 |             7938 |              0        |              11 |                    0 |                36 |               0.916667 |        158930 |        0.000566287 |             5897 |              0.50195  |                          1507 |                           0.100199 |                              21 |                             0.571429 |                   652 |                  0.0368098 |
|       20 |                  2513 |     1966 |      0.641404 |    128027 |    1           |              10293 |               1         |        231073 |         1          |                  5499 |                   1        |               5796 |                1        |             7938 |              0.500504 |              11 |                    1 |                36 |               1        |        158930 |        1           |             5897 |              0.505172 |                          1507 |                           1        |                              21 |                             1        |                   652 |                  1         |
|       40 |                  5200 |     1966 |      0.641404 |    128027 |    1           |              10293 |               1         |        231073 |         1          |                  5499 |                   1        |               5796 |                1        |             7938 |              0.500504 |              11 |                    1 |                36 |               1        |        158930 |        1           |             5897 |              0.505172 |                          1507 |                           1        |                              21 |                             1        |                   652 |                  1         |
|       80 |                  9789 |     1966 |      0.641404 |    128027 |    1           |              10293 |               1         |        231073 |         1          |                  5499 |                   1        |               5796 |                1        |             7938 |              0.997732 |              11 |                    1 |                36 |               1        |        158930 |        1           |             5897 |              0.505172 |                          1507 |                           1        |                              21 |                             1        |                   652 |                  1         |

PortScan recall: 0.0233% at n_bins=6.0 -> 100.0000% at n_bins=80.0.


### Bin-edge tied-mass saturation check (top 15 by tied mass, at n_bins=6 fit)

| feature                    |   edge |      n |   tied_count |   tied_mass |
|:---------------------------|-------:|-------:|-------------:|------------:|
| syn_without_synack_count   |      0 | 482016 |       482016 |  1          |
| bwd_pkt_len_mean           |      0 | 482016 |       127697 |  0.264923   |
| pkt_len_range              |      0 | 482016 |       121327 |  0.251707   |
| distinct_dst_ports_per_src |     66 | 482016 |        48588 |  0.100802   |
| distinct_dst_ports_per_src |    357 | 482016 |        37728 |  0.0782713  |
| pkt_len_range              |     16 | 482016 |        34204 |  0.0709603  |
| distinct_dst_ports_per_src |     63 | 482016 |        27400 |  0.0568446  |
| flows_per_src              |  48656 | 482016 |        22789 |  0.0472785  |
| flows_per_src              |  33457 | 482016 |        20646 |  0.0428326  |
| flows_per_src              |  26756 | 482016 |        16666 |  0.0345756  |
| distinct_dst_ports_per_src |     58 | 482016 |        16060 |  0.0333184  |
| distinct_dst_ports_per_src |     74 | 482016 |        14272 |  0.029609   |
| flows_per_src              |  22447 | 482016 |        11226 |  0.0232897  |
| pkt_len_range              |     54 | 482016 |         2679 |  0.00555791 |
| bwd_pkt_len_mean           |     86 | 482016 |         2060 |  0.00427372 |

## Headline: three-priority policy at 1.00% budget, per-day renewing meter

- Total flows: 2,830,743
- Escalated: 20,976 (0.7410%)
- Per priority (admitted): P1=2,827, P2=15,030, P3=3,119
- Pre-meter "fires first" waterfall: P1=561,249, P2 (of what P1 didn't take)=573,136, P3 (of what neither took)=3,306
- Attack flows captured: 144/557,646 (0.0258%), missed: 557,502
- Precision of escalated set: 0.6865%
- Benign FPR of escalated set: 0.9165%

### Per-class capture/miss at the headline operating point

| label                      |       n |   escalated |   missed |      recall |   caught_by_p1 |   caught_by_p2 |   caught_by_p3 |
|:---------------------------|--------:|------------:|---------:|------------:|---------------:|---------------:|---------------:|
| BENIGN                     | 2273097 |       20832 |  2252265 | 0.00916459  |           2826 |          15030 |           2976 |
| Bot                        |    1966 |           4 |     1962 | 0.00203459  |              0 |              0 |              4 |
| DDoS                       |  128027 |           0 |   128027 | 0           |              0 |              0 |              0 |
| DoS GoldenEye              |   10293 |           1 |    10292 | 9.71534e-05 |              0 |              0 |              1 |
| DoS Hulk                   |  231073 |         130 |   230943 | 0.000562593 |              0 |              0 |            130 |
| DoS Slowhttptest           |    5499 |           1 |     5498 | 0.000181851 |              0 |              0 |              1 |
| DoS slowloris              |    5796 |           1 |     5795 | 0.000172533 |              0 |              0 |              1 |
| FTP-Patator                |    7938 |           0 |     7938 | 0           |              0 |              0 |              0 |
| Heartbleed                 |      11 |           0 |       11 | 0           |              0 |              0 |              0 |
| Infiltration               |      36 |           0 |       36 | 0           |              0 |              0 |              0 |
| PortScan                   |  158930 |           3 |   158927 | 1.88762e-05 |              1 |              0 |              2 |
| SSH-Patator                |    5897 |           4 |     5893 | 0.000678311 |              0 |              0 |              4 |
| Web Attack – Brute Force   |    1507 |           0 |     1507 | 0           |              0 |              0 |              0 |
| Web Attack – Sql Injection |      21 |           0 |       21 | 0           |              0 |              0 |              0 |
| Web Attack – XSS           |     652 |           0 |      652 | 0           |              0 |              0 |              0 |

### Why headline recall is so far below Priority 1's own unmetered per-class recall

`results/p1_rules.md` shows Priority 1 alone (no meter) gets 74-99% unmetered recall on DoS Hulk/GoldenEye/Slowloris/Slowhttptest/DDoS/PortScan/FTP-Patator — the rules clearly CAN fire on these classes. The near-zero headline numbers above are a **verified budget/ordering artifact, not a rule-coverage gap** (checked directly, per the brief's own instruction to rule this out before calling a 0% a miss):

- Priority 1's own reserved cap is small by construction: the brief requires extracting **all 62** attack-leaf rules ("do not hand-pick"), which match 19.83% of the whole pool — nowhere near the PDF's own worked example (6 hand-selected, highly selective rules matching ~0.096-0.1% of flows). Under the PDF's own illustrative 10/70/20 budget split, Priority 1's guaranteed share is `global_capacity - reserved(P2) - reserved(P3)` ≈ 10% of the 1% total budget ≈ 0.1% of a day's flows — a small fixed ceiling that a 62-rule, 19.83%-of-traffic match set blows through almost immediately.
- **Directly checked on Wednesday (the DoS Hulk day)**: n=692,703, Priority 1's cap ≈692 flows. Priority 1 matches 204,623 rows that day, but the **first attack row (chronologically) sits at position 72,871 — Priority 1's 692-flow cap is already exhausted by row 24,293, using 692/692 BENIGN false-positive matches, 0 of them attacks.** DoS Hulk's attack window starts well after Priority 1 has nothing left to admit that day. Per-day detail for every day:


| day                   |   n_flows |   p1_cap |   n_p1_matches_total |   first_attack_row_position |   p1_cap_exhausted_at_row_position | cap_exhausted_before_first_attack   |   n_attacks_among_admitted_p1_slice |
|:----------------------|----------:|---------:|---------------------:|----------------------------:|-----------------------------------:|:------------------------------------|------------------------------------:|
| monday                |    529918 |      529 |                17655 |                         nan |                              12997 | False                               |                                   0 |
| tuesday               |    445909 |      445 |                24161 |                       46207 |                              13762 | True                                |                                   0 |
| wednesday             |    692703 |      692 |               204623 |                       72871 |                              24293 | True                                |                                   0 |
| thursday_webattacks   |    170366 |      170 |                 6155 |                       12637 |                               9793 | True                                |                                   0 |
| thursday_infiltration |    288602 |      288 |                 9635 |                       66383 |                              11957 | True                                |                                   0 |
| friday_morning        |    191033 |      191 |                 5466 |                       24072 |                               7541 | True                                |                                   0 |
| friday_portscan       |    286467 |      286 |               162170 |                        1463 |                              12652 | False                               |                                   1 |
| friday_ddos           |    225745 |      225 |               131384 |                       18883 |                              12519 | True                                |                                   0 |

This is a structural consequence of evaluating an unfiltered, brief-mandated 62-rule Priority 1 inside a budget/priority framework the PDF itself designed around a small, coverage-maximizing SELECTED subset (its own Step 1: "evaluate all 71 attack-leaf rules ... select the subset that maximizes attack-flow coverage under the 1% transfer constraint") — a step this evaluation deliberately does not perform, per this task's own explicit instruction to extract every rule rather than hand-pick. Not re-tuned or re-selected to improve this number, per the honesty requirements.

## Unseen-class recall — the real test (reported even where bad)

Bot/Infiltration are included for completeness but, per the discrepancy above, Priority 1 CAN in principle name them (they have tree leaves) — they are not a clean zero-day test the way Heartbleed and the Web Attack sub-variants are.

| label                      |    n | p1_has_leaf_for_this_class   |   caught_by_p1 |   caught_by_p2 |   caught_by_p3 |   total_escalated |     recall |
|:---------------------------|-----:|:-----------------------------|---------------:|---------------:|---------------:|------------------:|-----------:|
| Bot                        | 1966 | True                         |              0 |              0 |              4 |                 4 | 0.00203459 |
| Infiltration               |   36 | True                         |              0 |              0 |              0 |                 0 | 0          |
| Heartbleed                 |   11 | False                        |              0 |              0 |              0 |                 0 | 0          |
| Web Attack – Brute Force   | 1507 | False                        |              0 |              0 |              0 |                 0 | 0          |
| Web Attack – XSS           |  652 | False                        |              0 |              0 |              0 |                 0 | 0          |
| Web Attack – Sql Injection |   21 | False                        |              0 |              0 |              0 |                 0 | 0          |

## Budget sweep — 0.01% to 10%, stacked by priority

|   budget_fraction |     n_total |   attack_coverage |   benign_fpr |   escalated_total |   priority1_escalated |   priority2_escalated |   priority3_escalated |
|------------------:|------------:|------------------:|-------------:|------------------:|----------------------:|----------------------:|----------------------:|
|            0.0001 | 2.83074e+06 |       0           |  0.000126259 |               287 |                    23 |                   220 |                    44 |
|            0.0005 | 2.83074e+06 |       0           |  0.000556949 |              1266 |                   134 |                   932 |                   200 |
|            0.001  | 2.83074e+06 |       1.79325e-06 |  0.00102899  |              2340 |                   275 |                  1673 |                   392 |
|            0.005  | 2.83074e+06 |       1.97258e-05 |  0.00473935  |             10784 |                  1410 |                  7610 |                  1764 |
|            0.01   | 2.83074e+06 |       0.000258228 |  0.00916459  |             20976 |                  2827 |                 15030 |                  3119 |
|            0.02   | 2.83074e+06 |       0.000450106 |  0.0169729   |             38832 |                  5656 |                 29870 |                  3306 |
|            0.05   | 2.83074e+06 |       0.00322606  |  0.0396107   |             91838 |                 14145 |                 74387 |                  3306 |
|            0.1    | 2.83074e+06 |       0.00982523  |  0.0768603   |            180190 |                 28301 |                148583 |                  3306 |

![load vs coverage](figures/load_vs_coverage.png)


**Does this beat the existing selector's <=0.4% pooled attack recall at <=1% escalation (STATUS.md)?** At budget_fraction=0.01, attack coverage here is 0.0258%. **No, this does not clear that bar.**

## Cross-day: fit Monday+Tuesday (full), evaluate Wednesday/Thursday/Friday only

Fit: 961,992 Monday+Tuesday benign rows (full days, not chronologically halved — the day boundary itself is the fit/eval split here). Eval: 1,854,916 rows across Wed/Thu/Fri, never touched by fitting.

- Escalated: 17,983 (0.9695%)
- Per priority (admitted): P1=1,851, P2=14,840, P3=1,292
- Attack coverage: 0.0257% (140/543,811)
- Precision: 0.7785%
- Benign FPR: 1.3609%

### Cross-day per-class capture/miss

| label                      |       n |   escalated |   missed |      recall |   caught_by_p1 |   caught_by_p2 |   caught_by_p3 |
|:---------------------------|--------:|------------:|---------:|------------:|---------------:|---------------:|---------------:|
| BENIGN                     | 1311105 |       17843 |  1293262 | 0.0136091   |           1850 |          14840 |           1153 |
| Bot                        |    1966 |           4 |     1962 | 0.00203459  |              0 |              0 |              4 |
| DDoS                       |  128027 |           0 |   128027 | 0           |              0 |              0 |              0 |
| DoS GoldenEye              |   10293 |           1 |    10292 | 9.71534e-05 |              0 |              0 |              1 |
| DoS Hulk                   |  231073 |         130 |   230943 | 0.000562593 |              0 |              0 |            130 |
| DoS Slowhttptest           |    5499 |           1 |     5498 | 0.000181851 |              0 |              0 |              1 |
| DoS slowloris              |    5796 |           1 |     5795 | 0.000172533 |              0 |              0 |              1 |
| Heartbleed                 |      11 |           0 |       11 | 0           |              0 |              0 |              0 |
| Infiltration               |      36 |           0 |       36 | 0           |              0 |              0 |              0 |
| PortScan                   |  158930 |           3 |   158927 | 1.88762e-05 |              1 |              0 |              2 |
| Web Attack – Brute Force   |    1507 |           0 |     1507 | 0           |              0 |              0 |              0 |
| Web Attack – Sql Injection |      21 |           0 |       21 | 0           |              0 |              0 |              0 |
| Web Attack – XSS           |     652 |           0 |      652 | 0           |              0 |              0 |              0 |

# Sampled escalation run with per-flow ground-truth accounting

Random sample of 300,000 flows drawn uniformly (no stratification) from the pooled 2,830,743-row CSVs, then shuffled into a fresh random permutation. `random_state=42` throughout, reported here for reproducibility.

**Shuffled order measures rule quality, not deployment behaviour.** A real switch sees chronological order, where Monday's pure-benign traffic arrives first and fills the meter before any attack ever appears (see the P1-starvation diagnostic above, measured on the real chronological pool) — this section deliberately removes that ordering effect to isolate how good the rules/signature-table/sampling themselves are.

## Sample composition

- Total: 300,000
- BENIGN: 240,577
- Attack: 59,423 (19.8077% of sample)

Per-class attack counts:


- Bot: 195
- DDoS: 13,722
- DoS GoldenEye: 1,079
- DoS Hulk: 24,559
- DoS Slowhttptest: 590
- DoS slowloris: 601
- FTP-Patator: 860
- Infiltration: 4
- PortScan: 16,981
- SSH-Patator: 602
- Web Attack – Brute Force: 167
- Web Attack – Sql Injection: 2
- Web Attack – XSS: 61


## Ground truth isolation

The `Label` column is used only to SCORE the policy's decisions below, never as an input to any of them — enforced by construction (Priority 1's rules, Priority 2's signature, and Priority 3's hash all read only the 10 tree features / flow identifier, never `Label`) and checked directly by `tests/test_label_isolation.py`.

## Output 2 — summary table (qualified vs admitted; the gap is the meter's effect)

| priority   |   qualified |   admitted |   attacks_admitted |   benign_admitted |   precision |
|:-----------|------------:|-----------:|-------------------:|------------------:|------------:|
| P1         |       59769 |        300 |                259 |                41 |  0.863333   |
| P2         |       60989 |       2400 |                  7 |              2393 |  0.00291667 |
| P3         |         352 |        300 |                 16 |               284 |  0.0533333  |
| overall    |      121110 |       3000 |                282 |              2718 |  0.094      |

`results/escalated_flows.csv` carries the full per-flow dump behind these totals (3,000 rows).


### Per attack class

| label                      |     n |   admitted |   caught_by_p1 |   caught_by_p2 |   caught_by_p3 |   fraction_of_class_admitted |
|:---------------------------|------:|-----------:|---------------:|---------------:|---------------:|-----------------------------:|
| Bot                        |   195 |          8 |              0 |              7 |              1 |                   0.0410256  |
| DDoS                       | 13722 |         85 |             85 |              0 |              0 |                   0.00619443 |
| DoS GoldenEye              |  1079 |          3 |              3 |              0 |              0 |                   0.00278035 |
| DoS Hulk                   | 24559 |         92 |             79 |              0 |             13 |                   0.00374608 |
| DoS Slowhttptest           |   590 |          5 |              5 |              0 |              0 |                   0.00847458 |
| DoS slowloris              |   601 |          0 |              0 |              0 |              0 |                   0          |
| FTP-Patator                |   860 |          1 |              1 |              0 |              0 |                   0.00116279 |
| Infiltration               |     4 |          0 |              0 |              0 |              0 |                   0          |
| PortScan                   | 16981 |         85 |             85 |              0 |              0 |                   0.00500559 |
| SSH-Patator                |   602 |          3 |              1 |              0 |              2 |                   0.00498339 |
| Web Attack – Brute Force   |   167 |          0 |              0 |              0 |              0 |                   0          |
| Web Attack – Sql Injection |     2 |          0 |              0 |              0 |              0 |                   0          |
| Web Attack – XSS           |    61 |          0 |              0 |              0 |              0 |                   0          |

## Output 3 — Priority 1 rule quality, unmetered (whole sample, independent of budget)

- Union match precision: **87.4065%** (52,242 attack matches vs 7,527 benign matches, n=300,000)
- Priority 1's cap at this budget: **300** flows
- **4 of 62 rules have MORE benign matches alone than the entire P1 cap — those rules cannot pay for themselves under this budget no matter how the meter is otherwise split.**


Rules exceeding the cap on benign matches alone:


|   rule_id | predicted_class   |   benign_matches |   attack_matches |   n_matches |   precision | exceeds_p1_cap_alone   |
|----------:|:------------------|-----------------:|-----------------:|------------:|------------:|:-----------------------|
|         0 | DoS Hulk          |             4191 |              815 |        5006 |   0.162805  | True                   |
|        28 | Bot               |             1255 |              124 |        1379 |   0.0899202 | True                   |
|        11 | Web Attack        |             1122 |              247 |        1369 |   0.180424  | True                   |
|        35 | DoS Hulk          |              441 |              183 |         624 |   0.293269  | True                   |

### All 62 rules, sorted by benign matches (descending)

|   rule_id | predicted_class   |   benign_matches |   attack_matches |   n_matches |   precision | exceeds_p1_cap_alone   |
|----------:|:------------------|-----------------:|-----------------:|------------:|------------:|:-----------------------|
|         0 | DoS Hulk          |             4191 |              815 |        5006 |   0.162805  | True                   |
|        28 | Bot               |             1255 |              124 |        1379 |   0.0899202 | True                   |
|        11 | Web Attack        |             1122 |              247 |        1369 |   0.180424  | True                   |
|        35 | DoS Hulk          |              441 |              183 |         624 |   0.293269  | True                   |
|         9 | PortScan          |              125 |            16812 |       16937 |   0.99262   | False                  |
|         1 | DoS Hulk          |               89 |              861 |         950 |   0.906316  | False                  |
|        10 | Web Attack        |               54 |                9 |          63 |   0.142857  | False                  |
|        29 | DoS GoldenEye     |               43 |              254 |         297 |   0.855219  | False                  |
|        14 | DoS Slowhttptest  |               40 |              298 |         338 |   0.881657  | False                  |
|         4 | DoS Slowhttptest  |               31 |                0 |          31 |   0         | False                  |
|        13 | PortScan          |               20 |               24 |          44 |   0.545455  | False                  |
|        12 | DoS slowloris     |               18 |              168 |         186 |   0.903226  | False                  |
|        30 | DoS slowloris     |               14 |                6 |          20 |   0.3       | False                  |
|        56 | DoS GoldenEye     |                8 |              471 |         479 |   0.983299  | False                  |
|        31 | DoS Slowhttptest  |                7 |                2 |           9 |   0.222222  | False                  |
|         5 | DoS Hulk          |                6 |               29 |          35 |   0.828571  | False                  |
|        17 | DoS Hulk          |                6 |                2 |           8 |   0.25      | False                  |
|        21 | FTP-Patator       |                6 |                0 |           6 |   0         | False                  |
|        55 | PortScan          |                6 |                2 |           8 |   0.25      | False                  |
|         3 | SSH-Patator       |                4 |                2 |           6 |   0.333333  | False                  |
|         7 | DoS GoldenEye     |                4 |               49 |          53 |   0.924528  | False                  |
|        18 | DDoS              |                4 |             5009 |        5013 |   0.999202  | False                  |
|        32 | Infiltration      |                4 |                0 |           4 |   0         | False                  |
|         2 | FTP-Patator       |                3 |              421 |         424 |   0.992925  | False                  |
|        16 | DoS Hulk          |                3 |                0 |           3 |   0         | False                  |
|        27 | FTP-Patator       |                3 |               82 |          85 |   0.964706  | False                  |
|        37 | DoS Slowhttptest  |                3 |              119 |         122 |   0.97541   | False                  |
|        41 | Infiltration      |                3 |                0 |           3 |   0         | False                  |
|        43 | Infiltration      |                3 |                1 |           4 |   0.25      | False                  |
|        26 | DoS Slowhttptest  |                2 |               24 |          26 |   0.923077  | False                  |
|        19 | DDoS              |                1 |               18 |          19 |   0.947368  | False                  |
|        22 | DoS Hulk          |                1 |                1 |           2 |   0.5       | False                  |
|        24 | FTP-Patator       |                1 |                1 |           2 |   0.5       | False                  |
|        25 | DoS Slowhttptest  |                1 |                2 |           3 |   0.666667  | False                  |
|        40 | SSH-Patator       |                1 |                3 |           4 |   0.75      | False                  |
|        52 | DoS Slowhttptest  |                1 |               19 |          20 |   0.95      | False                  |
|        53 | DoS Slowhttptest  |                1 |               14 |          15 |   0.933333  | False                  |
|        54 | DoS Slowhttptest  |                1 |                0 |           1 |   0         | False                  |
|        60 | DoS GoldenEye     |                1 |               27 |          28 |   0.964286  | False                  |
|         6 | DoS slowloris     |                0 |              119 |         119 |   1         | False                  |
|         8 | DoS GoldenEye     |                0 |              203 |         203 |   1         | False                  |
|        15 | DoS Hulk          |                0 |                2 |           2 |   1         | False                  |
|        20 | DoS Hulk          |                0 |               40 |          40 |   1         | False                  |
|        23 | FTP-Patator       |                0 |              337 |         337 |   1         | False                  |
|        33 | DoS Slowhttptest  |                0 |                7 |           7 |   1         | False                  |
|        34 | DoS slowloris     |                0 |              192 |         192 |   1         | False                  |
|        36 | DoS GoldenEye     |                0 |               42 |          42 |   1         | False                  |
|        38 | SSH-Patator       |                0 |                6 |           6 |   1         | False                  |
|        39 | SSH-Patator       |                0 |              288 |         288 |   1         | False                  |
|        42 | Web Attack        |                0 |               14 |          14 |   1         | False                  |
|        44 | PortScan          |                0 |                3 |           3 |   1         | False                  |
|        45 | PortScan          |                0 |                1 |           1 |   1         | False                  |
|        46 | DDoS              |                0 |             8655 |        8655 |   1         | False                  |
|        47 | DoS Hulk          |                0 |            14601 |       14601 |   1         | False                  |
|        48 | DoS Hulk          |                0 |                2 |           2 |   1         | False                  |
|        49 | DoS Hulk          |                0 |                1 |           1 |   1         | False                  |
|        50 | DoS Hulk          |                0 |               12 |          12 |   1         | False                  |
|        51 | DoS Hulk          |                0 |             1530 |        1530 |   1         | False                  |
|        57 | DoS Hulk          |                0 |               23 |          23 |   1         | False                  |
|        58 | DoS Hulk          |                0 |                0 |           0 | nan         | False                  |
|        59 | DoS Hulk          |                0 |               64 |          64 |   1         | False                  |
|        61 | Bot               |                0 |                1 |           1 |   1         | False                  |

## Output 4 — missed attacks

- Qualified but cut by the meter (fixable by reallocating budget): **52,076**
- Never flagged by any priority (not fixable by any budget change — the ten features cannot see these): **7,065**


### Cut by the meter, per class

| label                      |     n |
|:---------------------------|------:|
| Bot                        |   114 |
| DDoS                       | 13603 |
| DoS GoldenEye              |  1057 |
| DoS Hulk                   | 18083 |
| DoS Slowhttptest           |   519 |
| DoS slowloris              |   518 |
| FTP-Patator                |   840 |
| Infiltration               |     4 |
| PortScan                   | 16808 |
| SSH-Patator                |   300 |
| Web Attack – Brute Force   |   167 |
| Web Attack – Sql Injection |     2 |
| Web Attack – XSS           |    61 |

### Never flagged by any priority, per class (the more important group)

| label                      |    n |
|:---------------------------|-----:|
| Bot                        |   73 |
| DDoS                       |   34 |
| DoS GoldenEye              |   19 |
| DoS Hulk                   | 6384 |
| DoS Slowhttptest           |   66 |
| DoS slowloris              |   83 |
| FTP-Patator                |   19 |
| Infiltration               |    0 |
| PortScan                   |   88 |
| SSH-Patator                |  299 |
| Web Attack – Brute Force   |    0 |
| Web Attack – Sql Injection |    0 |
| Web Attack – XSS           |    0 |

**DoS Hulk dominates this group** (6,384 of 7,065, 90.4%) — the single largest concentration of attack flows the ten-feature policy structurally cannot see at all in this sample.


## Sanity checks

- **P3 precision vs attack fraction (label-independent hash should track its own eligible pool)**: n_p3_admitted=300, p3_precision=0.05333333333333334, whole_sample_attack_fraction=0.19807666666666668, p3_eligible_pool_attack_fraction (n=179242)=0.03951640798473572, z_score_vs_eligible_pool=1.2283958289079815, deviation_flagged=False
  P3's precision should track the ATTACK FRACTION OF ITS OWN ELIGIBLE POOL (flows P1 and P2 both declined), not the whole sample's -- P1 and P2 already skim off a disproportionate share of attacks (P1 alone: 87.4% precision, see Output 3) before P3 ever sees a flow, so the eligible pool is attack-DEPLETED relative to the whole sample by construction, not by any hash bias. Compared against the correct baseline (its own eligible pool), the observed precision is within 2 standard errors -- consistent with an unbiased hash, not evidence of one.

- **P3 admitted vs its own reserved cap (sampling spreads evenly, can't be front-loaded)**: n_p3_admitted=300, p3_cap=600, ratio_to_cap=0.5
  P3 admitted below its own 600-flow cap, but NOT because sampling is uneven: P1 (300/300 cap) and P2 (2,400/2,400 cap) both hit their own caps exactly, leaving only 300 of the global budget for P3 by the time it's evaluated -- the global meter, not P3's own reservation, is the binding constraint here. Separately, P3's own hash-qualified pool (flows P1/P2 declined AND the tau hash selected, n=352) is itself smaller than its 600-flow cap, since most of the 179,242 flows P1/P2 declined were never tau-selected in the first place.


## Honesty notes

- Every number above states its own n/denominator inline.
- `tau`, bin count, and the 10/70/20 budget split are unchanged from the main per-day analysis — none were tuned toward any target for this sample.
- Benign escalations under Priority 2 and Priority 3 are expected and correct by design, not a defect: Priority 2 is fitted on benign traffic only and has no notion of what an attack looks like, and Priority 3 samples without reading the flow at all. Only Priority 1's benign matches are a real, reducible cost (they consume a rule's own decision, which does encode attack structure).
- `tree.txt` remains the ground truth for the feature set (see `eval/parse_tree.py`).

# Selected sample for the agent pipeline (n=20)

Drawn from `results/escalated_flows.csv` (3,000 escalated flows: 282 attacks, 2,718 benign). `random_state=20`, reported here for reproducibility. Selection only -- no agent was run for this task.

**The 40% attack rate in this 20-record sample is NOT the natural rate.** Escalated flows are 282/3000 = 9.4% attacks; a natural, unweighted draw of 20 would carry about 2 attacks and 18 benign. This sample deliberately over-represents attacks (12 of 20) so the 7 rare attack classes have a chance to appear at all -- it measures whether the agents can DISCRIMINATE attack-shaped records from benign ones, not how the pipeline would perform at deployment base rates. Any accuracy figure computed from these 20 records must carry that caveat every time it's quoted.

## Attack class breakdown of the 12

- Bot: 2
- DDoS: 2
- DoS GoldenEye: 2
- DoS Hulk: 2
- DoS Slowhttptest: 2
- FTP-Patator: 1
- PortScan: 1


7 of 8 attack classes present in the 282-attack pool are represented here (at most 2 per class, alphabetical first pass, largest-class top-up if a pass doesn't fill the quota -- see `eval/select_agent_sample.py::select_attack_flows`).

## Priority breakdown of all 20

- P1: 10
- P2: 9
- P3: 1


Within the 12 attacks specifically: P1=10, P2=2, P3=0. **The 12 are P1-heavy** (10/12 admitted by Priority 1) -- this mirrors the source pool, where 259/282 (91.8%) of escalated attacks were caught by P1's rules, not P2 or P3. A consequence for interpreting agent results: most attack records will carry a `class_predicted` hint (see below), so this sample is better read as a test of whether agents corroborate or override a rule's own guess, not as a blind detection test from raw features alone -- the P2/P3-admitted attacks in this specific 20 (Bot) are the closest thing here to a blind test.

## `class_predicted` hint

**10 of 20 records carry a `class_predicted` field** (present only for Priority-1-admitted records, per the brief). This is the TREE'S OWN PREDICTION from its matched rule, not the ground-truth label -- but it stays in the record because the brief specifies it should, while flagging plainly that for these records the agent is effectively being told "a rule thinks this is <class>" before it ever looks at the raw features. Any strong agreement between agent output and `class_predicted` on these records is not independent corroboration; it may just be the agent reading the hint.

Record IDs carrying the hint: 145e00165f3840a78cd1690f14e4ec94, 1b21af7fb28e43f1a638f5686ba61c8c, 6846713a334440c1b9748d47e85b6ee4, 2948aa49825c4db489f8ef138bebe642, 405408a974e64a7381e2f8964fd08894, 7991405abc3349dbad44252e6c8832ab, 5b397707248e4b44b25c079b1157e6b5, 49bcc3b140234b459443572c7dab870b, 7437f3e358364328a9ca5e268f135e6a, 3aedc4e3b08245fc900074e601aee148

## The 20 records

| record_id | priority | admitted_by | attack class | has class_predicted hint |
|---|---|---|---|---|
| 145e00165f3840a78cd1690f14e4ec94 | 1 | P1 | DoS GoldenEye | yes |
| 1b21af7fb28e43f1a638f5686ba61c8c | 1 | P1 | PortScan | yes |
| 6846713a334440c1b9748d47e85b6ee4 | 1 | P1 | DoS Hulk | yes |
| 2948aa49825c4db489f8ef138bebe642 | 1 | P1 | DoS Hulk | yes |
| 405408a974e64a7381e2f8964fd08894 | 1 | P1 | FTP-Patator | yes |
| 2b7954cb1b324087a61cf3b140f754ab | 2 | P2 | Bot |  |
| 5db4f0dca10b4c62a5c79a2f788d548a | 2 | P2 |  |  |
| 7991405abc3349dbad44252e6c8832ab | 1 | P1 | DoS Slowhttptest | yes |
| 2031095dbd71481e9aea8c718e953f57 | 2 | P2 |  |  |
| f56d6007866140e5ad3e7e3be0f448be | 2 | P2 |  |  |
| 5b397707248e4b44b25c079b1157e6b5 | 1 | P1 | DoS GoldenEye | yes |
| 4d33af2908a54879869307416a6ffb43 | 2 | P2 | Bot |  |
| 49bcc3b140234b459443572c7dab870b | 1 | P1 | DoS Slowhttptest | yes |
| e209f4618ef8477e990f6079d0ad09a5 | 2 | P2 |  |  |
| e9324f383377462298da8f8dba0c04c9 | 3 | P3 |  |  |
| d81abee130e942bca60a021fcbb4cfa4 | 2 | P2 |  |  |
| 03c6f14dc5ac4f898a3ce677efaf9df1 | 2 | P2 |  |  |
| 7437f3e358364328a9ca5e268f135e6a | 1 | P1 | DDoS | yes |
| 3aedc4e3b08245fc900074e601aee148 | 1 | P1 | DDoS | yes |
| 5658142a21c74ecd9cbe075b160d1184 | 2 | P2 |  |  |


## Scoring methodology for the run that follows (not computed here -- no agent has run yet)

Agents will output a continuous `benign_plausibility` in [0,1], not a label. Scoring thresholds it at the project's derived verdict boundary, **0.3** (not tuned to these 20 records, and not to be tuned to maximise accuracy on them after the fact). Once real agent output exists, report:
- the confusion matrix (predicted attack/benign at the 0.3 threshold vs. `agent_key_20.csv`'s `is_attack`), with counts, not just percentages, and a confidence interval on any derived rate (n=20 is small enough that this matters a great deal);
- `benign_plausibility`'s distribution for the true-attack group and the true-benign group reported SEPARATELY (e.g. min/median/max or a small histogram for each) -- the separation between the two distributions is more informative at this n than accuracy at any single threshold, and should be reported even if the 0.3 threshold happens to land badly;
- explicit acknowledgement that a P1-heavy attack set (see above) makes any confusion-matrix cell hard to attribute cleanly to "the agent detected the attack" versus "the agent read the hint".

## Honesty notes

- n=20 supports almost nothing on its own -- every count above is reported as a count, not a percentage, except where a percentage is explicitly qualified by its denominator.
- Records were not reselected after seeing any result; this selection runs once, from a fixed `random_state=20`, and the two output files are final.
- Every number in this section states its own n.

