# Time and Ordering in Distributed Systems

Time is a fundamental challenge in distributed systems. Different nodes have different clocks, and there's no global time reference.

## The Problem with Physical Clocks

### Clock Drift
Every clock drifts at a slightly different rate.
- Quartz clocks drift ~1 second per month
- Atomic clocks are more accurate but expensive

### Clock Synchronization
NTP (Network Time Protocol) can synchronize clocks to within milliseconds.
- Not precise enough for strict ordering
- Network delays add uncertainty

## Logical Clocks

### Lamport Clocks
Introduced by Leslie Lamport in 1978.

**Rules:**
1. Before each event, increment the clock
2. When sending a message, attach the clock value
3. When receiving, set clock to max(local, received) + 1

**Limitation:** Cannot determine if events are concurrent.

### Vector Clocks
Track causality precisely.
- Each node maintains a vector of counts
- Can determine causality and concurrency

**Example:**
- Node A: [1, 0, 0]
- Node B: [1, 1, 0]
- Node C: [0, 0, 1]

If all elements of V1 ≤ V2, then V1 happened before V2.
Otherwise, events are concurrent.

## Ordering Guarantees

### Total Order
All nodes agree on the same order.
- Achieved through consensus (Raft, Paxos)
- More coordination, lower throughput

### Partial Order
Only causally related events are ordered.
- Preserves program order within each node
- Allows more concurrency

### Causal Order
If A causes B, A is ordered before B.
- Captures "happens-before" relationship
- Weaker than total order, stronger than partial

## Real-World Applications

### Conflict Resolution in CRDTs
Conflict-free Replicated Data Types use vector clocks.
- Deterministic merge operations
- No coordination required

### Snapshot Isolation
Database isolation level using timestamps.
- Each transaction sees consistent snapshot
- Allows concurrent reads

### Distributed Debugging
Vector clocks help trace causality across nodes.
- Reconstruct event order
- Find race conditions

## Connection to Consensus

Total ordering requires consensus:
- Leader assigns sequence numbers
- All nodes agree on the sequence

This connects time/ordering to the CAP theorem trade-offs.
