# LCHQMDriver

Superconducting-qubit calibration system for Quantum Machines OPX1000 hardware (MW-FEM + LF-FEM),
built on the qm-qua → QUAM → QUAlibrate stack. It mixes vendored official `qua-libs` calibration
nodes with this lab's custom `LCH_*` nodes, and is the QM backend for the vendor-neutral `scqo`
experiment API (`customized/scqo/`). New experiments are scqo-only by default — fused files in
`customized/scqo/experiments/`, no qualibrate node.

See [CLAUDE.md](CLAUDE.md) for the full architecture, conventions, and operating rules.
