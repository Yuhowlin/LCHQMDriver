"""Generate the QM vendor config (wiring.json + state.json) for the 6-qubit chip.

This is the ``generate_quam.py`` step of the QUAM workflow (see README.md):
it maps the sample ROSTER onto ``qualang_tools.wirer`` calls, then
``build_quam_wiring`` + ``build_quam`` materialise the vendor state.

Roster (components.toml schema 3) this config realises:
    [modes]   q1,q2,q6 = transmon        (fixed frequency, no z line)
              q3,q4,q5 = flux_transmon   (flux tunable, z line)
    [lines]   feedline: readout = q1..q6 -> ONE multiplexed feedline
              xy1..xy6                   -> one drive line per qubit
              z3,z4,z5                   -> flux line for q3,q4,q5
    (no [composites] -> no qubit_pairs / couplers)

Port map (controller 1; readout+xy on the MW-FEM in slot 7, flux on the
LF-FEM in slot 1):
    readout   : mw slot 7, in_port=1, out_port=1   (all six multiplexed here)
    xy  q1..q6: mw slot 7, out_port = 2,3,4,5,6,7
    z   q3,q4,q5: lf slot 1, out_port = 1,2,3

Safety: the build SAVES to ``STATE_DIR`` below (a dedicated folder), set via the
``QUAM_STATE_PATH`` env var *before* any QUAM load/save, so it can never overwrite
the live ``quam_state/``. Point ``STATE_DIR`` at the real setup folder when ready.

Run:
    python quam_config/generate_quam.py
"""

import os
from pathlib import Path

# --- output folder: keep the fresh 6Q config OUT of the live quam_state/ ------
# Must be set before importing/constructing Quam so save()/load() resolve here.
STATE_DIR = Path(__file__).resolve().parents[1] / "quam_state_6q"
os.environ["QUAM_STATE_PATH"] = str(STATE_DIR)

import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec, lf_fem_spec
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_config import Quam

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "10.21.30.201"  # QOP IP address        -- TODO: confirm for this cluster
port = 9516  # QOP port                            -- TODO: confirm for this cluster
cluster_name = "QPX1000_6"  # cluster name         -- TODO: confirm for this cluster

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
CON = 1  # controller index
MW_SLOT = 7  # MW-FEM: readout + xy drive
LF_SLOT = 1  # LF-FEM: flux

instruments = Instruments()
instruments.add_mw_fem(controller=CON, slots=[MW_SLOT])
instruments.add_lf_fem(controller=CON, slots=[LF_SLOT])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = [1, 2, 3, 4, 5, 6]  # q1..q6
flux_qubits = [3, 4, 5]  # flux_transmon; q1,q2,q6 are fixed transmons (no z)
# No qubit_pairs: the roster declares no couplers.

########################################################################################################################
# %%                                 Define the hardcoded channel addresses (the ports)
########################################################################################################################
# Readout: all six qubits multiplexed on ONE MW port (in=out=1).
res_ch = mw_fem_spec(con=CON, slot=MW_SLOT, in_port=1, out_port=1)

# XY drive ports on the MW-FEM: q1->2, q2->3, q3->4, q4->5, q5->6, q6->7.
xy_out_port = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
xy_ch = {q: mw_fem_spec(con=CON, slot=MW_SLOT, out_port=p) for q, p in xy_out_port.items()}

# Flux ports on the LF-FEM: q3->1, q4->2, q5->3.
flux_out_port = {3: 1, 4: 2, 5: 3}
flux_ch = {q: lf_fem_spec(con=CON, out_slot=LF_SLOT, out_port=p) for q, p in flux_out_port.items()}

########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# One shared/multiplexed readout feedline for all six qubits
connectivity.add_resonator_line(qubits=qubits, constraints=res_ch)
# One xy drive line per qubit
for q in qubits:
    connectivity.add_qubit_drive_lines(qubits=q, constraints=xy_ch[q])
# Flux lines for the flux-tunable qubits only
for q in flux_qubits:
    connectivity.add_qubit_flux_lines(qubits=q, constraints=flux_ch[q])

# Allocate the wiring (block_used_channels=False so the explicit constraints stand)
allocate_wiring(connectivity, instruments, block_used_channels=False)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=False)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
user_input = input(f"Do you want to save the QUAM to {STATE_DIR}? (y/n) ")
if user_input.lower() == "y":
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    machine = Quam()
    # Build the wiring (wiring.json) and initiate the QUAM
    build_quam_wiring(connectivity, host_ip, cluster_name, machine, port)

    # Reload QUAM, build the full QUAM object and save the state as state.json
    machine = Quam.load()
    build_quam(machine)
    print(f"Wrote wiring.json + state.json to {STATE_DIR}")
