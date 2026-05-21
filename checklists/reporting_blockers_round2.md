# Round-2 reporting blockers

Treat each item as a blocker before making claims about cartographic subset effectiveness.

- AddSent/AddOneSent rows are reported separately as all-row, original-only, adversarial-only, and paired metrics.
- `logs/table_audit.md` maps each table row to dataset path, hash, row count, run id, and metric source.
- Ranked equal-size subsets are not described as a disjoint partition.
- Region-pure subset diagnostics are reported when claims mention easy/ambiguous/hard regions.
- QA windowing is audited with `n_gold_records` and `n_non_gold_records`, or explicitly marked as legacy all-window cartography.
- Question-type mechanism features use the normalized taxonomy, not raw first tokens.
- Repeated random baselines are available before claiming selected subsets outperform random.
