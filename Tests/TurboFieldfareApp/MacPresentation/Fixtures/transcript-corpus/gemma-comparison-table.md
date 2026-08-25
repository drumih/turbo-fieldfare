### Concurrency Comparison

| Feature | GCD Queues | Swift Actors |
| :--- | :--- | :--- |
| **Model** | Low-level, closure-based. | High-level, language-level. |
| **Safety** | Manual management; prone to data races. | Built-in protection against data races. |
| **Mechanism** | Thread pools and dispatching. | Actor isolation and cooperative multitasking. |

* **GCD Queues**
    * Manual control over priority.
    * Risk of priority inversion.
* **Swift Actors**
    * Automatic synchronization.
    * Uses `await` to manage suspension points.

#### Task Checklist
- [x] Define data model
- [x] Implement synchronization
- [ ] Optimize performance

> "Concurrency is not just about running tasks in parallel, but about managing access to shared state safely."

[Apple Documentation: Swift Concurrency](https://developer.apple.com/documentation/swift/concurrency)

```swift
actor BankAccount {
    var balance = 0
    func deposit(amount: Int) { balance += amount }
}
```

The system uses $HOME to locate user files and $PATH to find executable binaries.