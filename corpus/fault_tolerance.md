# Fault Tolerance Patterns

Fault tolerance is the ability of a system to continue operating properly in the event of failures.

## Types of Faults

### Crash Faults
A node stops working completely. The simplest type of fault to handle.
- **Detection**: Heartbeat mechanisms
- **Recovery**: Node restart or failover

### Byzantine Faults
A node behaves arbitrarily, including maliciously. The hardest type to handle.
- **Detection**: Consensus among multiple nodes
- **Recovery**: Requires Byzantine Fault Tolerant (BFT) protocols

### Omission Faults
A node fails to send or receive messages.
- **Detection**: Timeouts and acknowledgments
- **Recovery**: Retransmission protocols

### Timing Faults
A node responds too slowly.
- **Detection**: Deadlines and monitoring
- **Recovery**: Timeout handling and request routing

## Recovery Strategies

### Checkpointing
Periodically saving system state to stable storage.
- Reduces recovery time after failure
- Trade-off between checkpoint frequency and performance overhead

### Write-Ahead Logging (WAL)
Recording operations before applying them.
- Enables replay of operations after crash
- Used in databases like PostgreSQL and SQLite

### Redundancy

Maintaining backup components that can take over when primary fails.

**Types:**
1. **Hot Standby**: Backup is running and synchronized
2. **Warm Standby**: Backup is running but not fully synchronized
3. **Cold Standby**: Backup is offline, started only when needed

## Circuit Breaker Pattern

Prevents cascade failures by stopping requests to a failing service.

**States:**
- **Closed**: Requests flow normally
- **Open**: Requests fail immediately
- **Half-Open**: Limited requests to test recovery

## Bulkhead Pattern

Isolates components to prevent failure propagation.
- Separates thread pools or connection pools
- Name comes from ship bulkheads that prevent flooding

## Relationship to Consensus

Fault tolerance often relies on consensus protocols:
- Leader election requires consensus
- Replicated state machines use consensus for ordering
- Configuration management needs consistent views

Without consensus, nodes might diverge and the system becomes inconsistent.
