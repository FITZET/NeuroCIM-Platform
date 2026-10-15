# Multi-agent task board

Updated: 2026-09-30

| Task ID | Requirement | Role | Allowed scope | Goal / acceptance | Status |
|---|---|---|---|---|---|
| R0-PLAN | All open requirements | Planner | Read-only | Selected three independent implementation candidates | VERIFIED |
| R0-QA | R-5.1, R-5.2, R-5.4 | QA | Read-only | 40 tests and retained integration evidence | VERIFIED |
| R0-COMP | R-3.3, R-4.1–R-4.4 | Release/compliance | Read-only | Found portability and redistribution blockers | VERIFIED |
| R1-CELL | R-2.1.1-B/C, R-3.3-B | Device/Core | `1_data_fitting/**`, `error_model.py`, their tests | Generalized cell CSV import while preserving uint8 path | VERIFIED |
| R1-GUI-INF | R-2.3-C/D/E, R-3.2-C | GUI/Application | `application.py`, `gui.py`, two GUI/application tests | Real LeNet single-sample float/ideal/device inference in worker | VERIFIED |
| R1-PORT | R-3.3-A/C, R-4.1, R-4.3 | Release/compliance | path defaults, environment scripts, notices, packaging manifest/tests | Remove personal runtime paths and define legal package boundary | VERIFIED |
| R2-RESNET-FORMAL | R-2.2-A/C, R-5.1/5.2 | Model/training | `3_models/Resnet20/**` | Formal reproducible training, full float evaluation and bounded differential evaluation | VERIFIED |
| R2-GUI-PPA | R-2.3-C/D, R-3.2-B | GUI/NeuroSim | `application.py`, `gui.py`, their tests/evidence | Background official NeuroSim PPA workflow with retained result and screenshot | VERIFIED |

| R3-GUI-TRAIN | R-2.3-D, R-2.4-C/D | GUI/Application | `application.py`, `gui.py`, optional `training_service.py`, dedicated tests/evidence | Real LeNet EAT background workflow with progress, cancellation and artifacts | VERIFIED |
| R3-BNN-FORMAL | R-2.2-A/C | Model/training | new `3_models/BNN/**` only | Reproducible BNN training, full MNIST evaluation and honest differential evidence | VERIFIED |

| R4-CELL-GUI | R-2.1.1-B/C/D, R-2.3-A/B | GUI/Application | `application.py`, `gui.py`, dedicated tests/evidence | Cell CSV preview and background fitting GUI using the verified fitting core | VERIFIED |
| R4-DELIVERY-DOC | R-4.4-A/B/C/D, R-5.4 | Release/docs | manuals, README index, documentation validation | Evidence-linked user, technical, environment and test manuals | VERIFIED |

| R5-PERF | R-3.2-C, R-5.3-A | Runtime/performance | `application.py`, runtime/benchmark modules and tests/evidence | Cached real single/batch inference meeting timing targets | VERIFIED |
| R5-ARRAY-GUI | R-2.3-B, R-3.2-A, R-5.2 | GUI/Core | `gui.py`, `mac_error_model.py`, dedicated tests/evidence | Array/bit controls drive real filtering plus theoretical linearity acceptance | VERIFIED |

## File ownership

Round 5 write scopes are disjoint. The supervisor exclusively owns this board,
`AUTOMATION_STATE.json`, `PROGRESS.md`, the requirements traceability document and
shared `__init__.py` exports.

## Gate

Rounds 1 through 5 are closed. The latest complete regression is 78 passed and one guarded formal-training test skipped; independent QA verified inference performance and array configuration.
