# Coding release engineering validation

Date: 2026-10-26. Profile: `v1.0-coding`. No official smoke or Full score is claimed.

The common runtime source is byte-identical to the reviewed clmemory source
snapshot 41ebcfb4467a037d519fa1e61a8e2a685769f195. Its existing 11-test engineering
suite passed on the deployed Linux native MemMachine stack. The suite contains
shared protocol fixtures; those are engineering checks, not cross-track research.
This export changes public documentation and registration scope only.

Six real GPT-4o-mini Coding-profile engineering fixtures passed: actual current
state versus unaccepted proposal; historical-state preservation; unauthorized
disclosure; injected instructions versus a legitimate historical test command;
questions about assistant proposals; and a separate actual-purchase case.
These synthetic fixtures test the generic filter contract; they are not CAMBench
or a Coding method-effectiveness evaluation. Earlier failed filter candidates
remain in private development receipts and are not presented as successes.

The Coding public HTTP path passed real text-embedding-v4/1024 and mini-filter
checks for health, authentication, synchronous Add, idempotency/conflict,
immediate Search, top_k=1/16/100, user isolation and invalid-top_k rejection.
The runtime has separate Coding source/index/cache/key/ledger state. Successful
compatible checks are reused without additional model calls for this packaging
change. Full capacity, official results and general filter effectiveness remain
unverified. The access-review mini budget is 200 cumulative physical requests
including failures; Add/Search concurrency starts at 1/1.
