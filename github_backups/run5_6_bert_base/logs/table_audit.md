# Table/evalset audit

This audit is generated from normalized metrics JSON files. It checks evalset names, example counts, dataset paths, and dataset hashes before tables are reported.

## Evalset identity checks

| evalset | expected n | observed n values | dataset paths | dataset hashes | status |
|---|---:|---|---|---|---|
| addonesent | 1787 | 1787 | data/qa/addonesent.jsonl | sha256:fe4b4d10a7d1a5afed58183078a4fe10820af5537d4e28a82cab990673fb0d6c | OK |
| addsent | 3560 | 3560 | data/qa/addsent.jsonl | sha256:698ce6f7c1779b1b83d253f1c1b3fc2058f278ec4e656dae445dc5c7a7de98df | OK |
| squad_dev | 10570 | 10570 | data/qa/squad_dev.jsonl | sha256:940197b8001a1309109a52f2d51d1b3eb25d1db49950b24705dc8641f1c19502 | OK |

## AddSent/AddOneSent label check

A likely label swap is flagged if AddSent and AddOneSent example counts do not match the expected adversarial SQuAD counts or if either evalset is backed by more than one dataset hash.

Conclusion: evalset labels are consistent with expected AddSent/AddOneSent example counts.
