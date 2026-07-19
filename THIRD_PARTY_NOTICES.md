# Third-party software and model terms

TurboFieldfare repository source is licensed under the
[Apache License 2.0](LICENSE). That license applies to TurboFieldfare's own source
and documentation. It does not relicense model weights or third-party
packages.

This file records the dependency review performed on 2026-07-15. It is an
attribution aid, not legal advice. Anyone distributing a compiled product must
also preserve the license and NOTICE material required by the exact dependency
versions included in that product.

## Model weights

Model weights are not included in this repository. The installer downloads a
pinned revision of
[`majentik/gemma-4-26B-A4B-TurboQuant-MLX-4bit`](https://huggingface.co/majentik/gemma-4-26B-A4B-TurboQuant-MLX-4bit)
and repacks it locally. The model card labels that checkpoint Apache-2.0 and
identifies `google/gemma-4-26B-A4B` as its base model. Google publishes Gemma 4
under the
[Apache License 2.0](https://ai.google.dev/gemma/apache_2).

Downloaded weights remain a separate work governed by their source terms. Do
not redistribute weights as part of TurboFieldfare releases.

## Swift package graph

The following table covers the complete graph reported by
`swift package show-dependencies` from the checked-in
[`Package.resolved`](Package.resolved). Exact revisions are recorded there.

| Package | Version | License in locked checkout |
| --- | --- | --- |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.3 | Apache-2.0 |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | 2.3.6 | Apache-2.0 |
| [swift-huggingface](https://github.com/huggingface/swift-huggingface) | 0.9.0 | Apache-2.0 |
| [EventSource](https://github.com/mattt/EventSource) | 1.4.1 | MIT |
| [swift-nio](https://github.com/apple/swift-nio) | 2.99.0 | Apache-2.0; upstream NOTICE applies |
| [swift-atomics](https://github.com/apple/swift-atomics) | 1.3.0 | Apache-2.0 with Runtime Library Exception |
| [swift-collections](https://github.com/apple/swift-collections) | 1.5.1 | Apache-2.0 with Runtime Library Exception |
| [swift-system](https://github.com/apple/swift-system) | 1.6.4 | Apache-2.0 with Runtime Library Exception |
| [swift-crypto](https://github.com/apple/swift-crypto) | 4.5.0 | Apache-2.0; upstream NOTICE applies |
| [swift-asn1](https://github.com/apple/swift-asn1) | 1.7.0 | Apache-2.0; upstream NOTICE applies |
| [yyjson](https://github.com/ibireme/yyjson) | 0.12.0 | MIT |

No copyleft or custom non-commercial license was found in this resolved graph.
The dependency license files remain authoritative. For binary distribution,
collect their license and NOTICE files from the exact revisions in
`Package.resolved`; do not treat this summary as a substitute for that bundle.
