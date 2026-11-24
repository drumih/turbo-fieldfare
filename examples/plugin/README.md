# Out-of-Tree Engine Plugin Example (`examples/plugin`)

This standalone SwiftPM package demonstrates how a third-party package can
depend on `TinyTitanLib`, implement an `EnginePlugin` (TPP/1,
`docs/plugin-api-v1.md`) in its own library target (`EchoPlugin`), register it
at startup via `EngineFactories.pluginRegistry.register(...)`, and resolve a
matching install without modifying any TinyTitan source file.

Run it directly or via the repository check script:

```bash
tools/plugin-dependency-check.sh
```
