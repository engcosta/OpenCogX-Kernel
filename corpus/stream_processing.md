# Stream Processing and Event Sourcing

Modern systems increasingly rely on streams of events rather than static databases.

## Event Sourcing

Store all changes as a sequence of events.

### Core Concepts
1. **Events are immutable**: Once recorded, never change
2. **Current state is derived**: Replay events to reconstruct state
3. **Audit trail built-in**: Complete history of changes

### Benefits
- Complete audit trail
- Temporal queries (state at any point in time)
- Easy debugging
- Supports CQRS (Command Query Responsibility Segregation)

### Challenges
- Event schema evolution
- Storage growth
- Query performance for current state

## Stream Processing

Process data as it arrives, not in batches.

### Key Concepts

#### Windowing
Grouping events by time or count.
- **Tumbling windows**: Fixed, non-overlapping
- **Sliding windows**: Fixed, overlapping
- **Session windows**: Dynamic, based on activity gaps

#### Watermarks
Track progress in event time.
- Handle late-arriving events
- Balance completeness vs latency

#### State Management
Maintaining state across events.
- Checkpointing for fault tolerance
- Partitioned state for scalability

## Technologies

### Apache Kafka
Distributed event streaming platform.
- Durable log storage
- Consumer groups for parallel processing
- Exactly-once semantics (with transactions)

### Apache Flink
Stream processing framework.
- Event time processing
- Stateful computations
- Exactly-once guarantees

### Apache Kafka Streams
Library for stream processing.
- Runs within your application
- Uses Kafka for everything

## Patterns

### Event-Driven Architecture
Components communicate via events.
- Loose coupling
- Asynchronous processing
- Better scalability

### CQRS
Separate read and write models.
- Write side: Event sourcing
- Read side: Optimized projections

### Saga Pattern
Coordinate long-running processes.
- Sequence of transactions
- Compensating actions for rollback

## Connection to Distributed Systems Concepts

### Ordering
Events need ordering guarantees.
- Kafka guarantees order within partitions
- Cross-partition ordering requires additional coordination

### Consistency
Eventually consistent reads.
- Consumers may lag behind producers
- Read-your-writes requires special handling

### Fault Tolerance
Must handle failures gracefully.
- Checkpointing state
- Reprocessing from offsets

### CAP Theorem
Stream systems typically choose AP:
- Available and partition tolerant
- Eventually consistent reads
