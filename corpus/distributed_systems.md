# Distributed Systems Fundamentals

A distributed system is a collection of autonomous computing elements that appears to its users as a single coherent system.

## Key Concepts

### CAP Theorem

The CAP theorem, also known as Brewer's theorem, states that a distributed system can only provide two of three guarantees simultaneously:

1. **Consistency**: Every read receives the most recent write or an error
2. **Availability**: Every request receives a (non-error) response, without guarantee that it contains the most recent write
3. **Partition Tolerance**: The system continues to operate despite an arbitrary number of messages being dropped (or delayed) by the network between nodes

In practice, since network partitions are unavoidable, systems must choose between consistency and availability.

### Consensus Protocols

Consensus is the process of agreeing on one result among a group of participants. This is fundamental to distributed systems.

#### Paxos
- Developed by Leslie Lamport
- Provides safety (consistency) even with network partitions
- Complex to implement correctly

#### Raft
- Designed as an understandable alternative to Paxos
- Uses leader election and log replication
- Widely adopted in modern systems (etcd, Consul)

### Replication

Replication ensures data is copied across multiple nodes for fault tolerance.

**Types:**
- **Synchronous**: Write waits for all replicas to acknowledge
- **Asynchronous**: Write returns immediately; replicas update later
- **Semi-synchronous**: Write waits for at least one replica

### Consistency Models

Different consistency guarantees trade off between performance and correctness:

1. **Linearizability**: Operations appear instantaneous
2. **Sequential Consistency**: Operations appear in some total order
3. **Causal Consistency**: Causally related operations appear in order
4. **Eventual Consistency**: Updates eventually propagate to all nodes

## Practical Implications

Systems like Apache Cassandra choose availability and partition tolerance, offering eventual consistency.
Systems like etcd or Zookeeper choose consistency and partition tolerance, sacrificing availability during partitions.

The choice depends on the application requirements. Banking systems typically require strong consistency, while social media feeds can tolerate eventual consistency.
