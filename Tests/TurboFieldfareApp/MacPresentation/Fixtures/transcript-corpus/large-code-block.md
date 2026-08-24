### Streaming buffer implementation

The full source of the ring buffer follows.

```swift
import Foundation

struct RingSlot000 {
    let index: Int = 0
    var payload: [UInt8] = Array(repeating: 0, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 7 }
}

struct RingSlot001 {
    let index: Int = 1
    var payload: [UInt8] = Array(repeating: 1, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 8 }
}

struct RingSlot002 {
    let index: Int = 2
    var payload: [UInt8] = Array(repeating: 2, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 9 }
}

struct RingSlot003 {
    let index: Int = 3
    var payload: [UInt8] = Array(repeating: 3, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 10 }
}

struct RingSlot004 {
    let index: Int = 4
    var payload: [UInt8] = Array(repeating: 4, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 11 }
}

struct RingSlot005 {
    let index: Int = 5
    var payload: [UInt8] = Array(repeating: 5, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 12 }
}

struct RingSlot006 {
    let index: Int = 6
    var payload: [UInt8] = Array(repeating: 6, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 13 }
}

struct RingSlot007 {
    let index: Int = 7
    var payload: [UInt8] = Array(repeating: 7, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 14 }
}

struct RingSlot008 {
    let index: Int = 8
    var payload: [UInt8] = Array(repeating: 8, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 15 }
}

struct RingSlot009 {
    let index: Int = 9
    var payload: [UInt8] = Array(repeating: 9, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 16 }
}

struct RingSlot010 {
    let index: Int = 10
    var payload: [UInt8] = Array(repeating: 10, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 17 }
}

struct RingSlot011 {
    let index: Int = 11
    var payload: [UInt8] = Array(repeating: 11, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 18 }
}

struct RingSlot012 {
    let index: Int = 12
    var payload: [UInt8] = Array(repeating: 12, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 19 }
}

struct RingSlot013 {
    let index: Int = 13
    var payload: [UInt8] = Array(repeating: 13, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 20 }
}

struct RingSlot014 {
    let index: Int = 14
    var payload: [UInt8] = Array(repeating: 14, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 21 }
}

struct RingSlot015 {
    let index: Int = 15
    var payload: [UInt8] = Array(repeating: 15, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 22 }
}

struct RingSlot016 {
    let index: Int = 16
    var payload: [UInt8] = Array(repeating: 16, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 23 }
}

struct RingSlot017 {
    let index: Int = 17
    var payload: [UInt8] = Array(repeating: 17, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 24 }
}

struct RingSlot018 {
    let index: Int = 18
    var payload: [UInt8] = Array(repeating: 18, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 25 }
}

struct RingSlot019 {
    let index: Int = 19
    var payload: [UInt8] = Array(repeating: 19, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 26 }
}

struct RingSlot020 {
    let index: Int = 20
    var payload: [UInt8] = Array(repeating: 20, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 27 }
}

struct RingSlot021 {
    let index: Int = 21
    var payload: [UInt8] = Array(repeating: 21, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 28 }
}

struct RingSlot022 {
    let index: Int = 22
    var payload: [UInt8] = Array(repeating: 22, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 29 }
}

struct RingSlot023 {
    let index: Int = 23
    var payload: [UInt8] = Array(repeating: 23, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 30 }
}

struct RingSlot024 {
    let index: Int = 24
    var payload: [UInt8] = Array(repeating: 24, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 31 }
}

struct RingSlot025 {
    let index: Int = 25
    var payload: [UInt8] = Array(repeating: 25, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 32 }
}

struct RingSlot026 {
    let index: Int = 26
    var payload: [UInt8] = Array(repeating: 26, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 33 }
}

struct RingSlot027 {
    let index: Int = 27
    var payload: [UInt8] = Array(repeating: 27, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 34 }
}

struct RingSlot028 {
    let index: Int = 28
    var payload: [UInt8] = Array(repeating: 28, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 35 }
}

struct RingSlot029 {
    let index: Int = 29
    var payload: [UInt8] = Array(repeating: 29, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 36 }
}

struct RingSlot030 {
    let index: Int = 30
    var payload: [UInt8] = Array(repeating: 30, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 37 }
}

struct RingSlot031 {
    let index: Int = 31
    var payload: [UInt8] = Array(repeating: 31, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 38 }
}

struct RingSlot032 {
    let index: Int = 32
    var payload: [UInt8] = Array(repeating: 32, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 39 }
}

struct RingSlot033 {
    let index: Int = 33
    var payload: [UInt8] = Array(repeating: 33, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 40 }
}

struct RingSlot034 {
    let index: Int = 34
    var payload: [UInt8] = Array(repeating: 34, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 41 }
}

struct RingSlot035 {
    let index: Int = 35
    var payload: [UInt8] = Array(repeating: 35, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 42 }
}

struct RingSlot036 {
    let index: Int = 36
    var payload: [UInt8] = Array(repeating: 36, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 43 }
}

struct RingSlot037 {
    let index: Int = 37
    var payload: [UInt8] = Array(repeating: 37, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 44 }
}

struct RingSlot038 {
    let index: Int = 38
    var payload: [UInt8] = Array(repeating: 38, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 45 }
}

struct RingSlot039 {
    let index: Int = 39
    var payload: [UInt8] = Array(repeating: 39, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 46 }
}

struct RingSlot040 {
    let index: Int = 40
    var payload: [UInt8] = Array(repeating: 40, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 47 }
}

struct RingSlot041 {
    let index: Int = 41
    var payload: [UInt8] = Array(repeating: 41, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 48 }
}

struct RingSlot042 {
    let index: Int = 42
    var payload: [UInt8] = Array(repeating: 42, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 49 }
}

struct RingSlot043 {
    let index: Int = 43
    var payload: [UInt8] = Array(repeating: 43, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 50 }
}

struct RingSlot044 {
    let index: Int = 44
    var payload: [UInt8] = Array(repeating: 44, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 51 }
}

struct RingSlot045 {
    let index: Int = 45
    var payload: [UInt8] = Array(repeating: 45, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 52 }
}

struct RingSlot046 {
    let index: Int = 46
    var payload: [UInt8] = Array(repeating: 46, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 53 }
}

struct RingSlot047 {
    let index: Int = 47
    var payload: [UInt8] = Array(repeating: 47, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 54 }
}

struct RingSlot048 {
    let index: Int = 48
    var payload: [UInt8] = Array(repeating: 48, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 55 }
}

struct RingSlot049 {
    let index: Int = 49
    var payload: [UInt8] = Array(repeating: 49, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 56 }
}

struct RingSlot050 {
    let index: Int = 50
    var payload: [UInt8] = Array(repeating: 50, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 57 }
}

struct RingSlot051 {
    let index: Int = 51
    var payload: [UInt8] = Array(repeating: 51, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 58 }
}

struct RingSlot052 {
    let index: Int = 52
    var payload: [UInt8] = Array(repeating: 52, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 59 }
}

struct RingSlot053 {
    let index: Int = 53
    var payload: [UInt8] = Array(repeating: 53, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 60 }
}

struct RingSlot054 {
    let index: Int = 54
    var payload: [UInt8] = Array(repeating: 54, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 61 }
}

struct RingSlot055 {
    let index: Int = 55
    var payload: [UInt8] = Array(repeating: 55, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 62 }
}

struct RingSlot056 {
    let index: Int = 56
    var payload: [UInt8] = Array(repeating: 56, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 63 }
}

struct RingSlot057 {
    let index: Int = 57
    var payload: [UInt8] = Array(repeating: 57, count: 64)
    func advance(by step: Int) -> Int { (index + step) % 64 }
}

```

The buffer wraps once the write cursor passes the capacity.

