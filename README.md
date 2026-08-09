# scqo-qm

The Quantum Machines OPX1000 backend for the vendor-neutral `scqo` experiment API
(`scqo_qm/` — one fused file per experiment), plus the vendored official `qua-libs`
qualibrate calibrations for GUI use. Built on the qm-qua → QUAM → QUAlibrate stack.
Renamed from LCHQMDriver in the v1 restructure; the custom LCH_* qualibrate shells are
retired (experiments run through `scqo run`).

See [CLAUDE.md](CLAUDE.md) for the full architecture, conventions, and operating rules.
