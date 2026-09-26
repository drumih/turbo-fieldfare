import SwiftUI

/// Selects Apple's property-wrapper type without expanding the SDK's State
/// macro, whose plugin is absent from some Command Line Tools installations.
/// Initial values are evaluated eagerly; keep expensive reference construction
/// outside view bodies and never combine an inline default with init assignment.
public typealias StoredState<Value> = SwiftUI.State<Value>
