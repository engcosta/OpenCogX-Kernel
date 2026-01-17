# Database Replication and Partitioning

Modern databases use replication and partitioning to achieve scalability and fault tolerance.

## Replication Strategies

### Single-Leader Replication
One node (leader) handles writes; replicas follow.

**Advantages:**
- Simple to implement
- No write conflicts

**Disadvantages:**
- Leader is a bottleneck
- Leader failure requires failover

### Multi-Leader Replication
Multiple nodes can accept writes.

**Advantages:**
- Higher write throughput
- Tolerance for datacenter failures

**Disadvantages:**
- Conflict resolution needed
- More complex implementation

### Leaderless Replication
Any node can accept reads and writes.

**Examples:** Amazon Dynamo, Cassandra, Riak

**Techniques:**
- Read repair: Fix stale data during reads
- Anti-entropy: Background synchronization
- Quorum reads/writes: Ensure consistency

## Partitioning (Sharding)

Dividing data across multiple nodes.

### Partitioning by Key Range
Each partition holds a range of keys.

**Advantages:**
- Efficient range queries
- Predictable distribution

**Disadvantages:**
- Hot spots if access is skewed

### Partitioning by Hash
Each partition covers a hash range.

**Advantages:**
- Even distribution
- No hot spots for key-based access

**Disadvantages:**
- Range queries inefficient
- Resharding is complex

### Consistent Hashing
Minimizes data movement when nodes are added/removed.
- Used in Cassandra, DynamoDB
- Virtual nodes improve balance

## Secondary Indexes

Challenges with partitioned secondary indexes:

### Local Indexes
Each partition maintains its own index.
- Write: Update one partition
- Read: Query all partitions (scatter-gather)

### Global Indexes
Index is partitioned separately from data.
- Write: May need to update multiple partitions
- Read: Query fewer partitions

## Transactions in Distributed Databases

### Two-Phase Commit (2PC)
Coordinator ensures all nodes commit or rollback.
- Prepare phase: Nodes vote
- Commit phase: Coordinator decides

**Problem:** Coordinator failure can block transactions.

### Three-Phase Commit (3PC)
Adds a pre-commit phase to reduce blocking.

### Saga Pattern
Long-running transactions as series of local transactions.
- Each step has a compensating action
- Eventually consistent

## Connection to Fault Tolerance

Replication provides redundancy for fault tolerance.
Partitioning enables horizontal scaling.
Both require careful handling of network partitions and node failures.
