# Requirements traceability

Source: `20260421-DRAM存算仿真平台外包开发规格书-zys(1).docx`  
Status date: 2026-09-30

Status vocabulary: **verified**, **implemented/unverified**, **partial**, **blocked**,
and **contract item**. A requirement is verified only when an executable test or
retained run artifact demonstrates it.

| ID | Requirement | Implementation / evidence | Status |
|---|---|---|---|
| R-2.1.1-A | Import cell multiplication CSV | `1_data_fitting`, `HardwareErrorModel`; four real device files | verified |
| R-2.1.1-B | Configurable column mapping/units | Cell importer supports configurable columns, units, encoding and delimiter; covered by tests | verified |
| R-2.1.1-C | 1/2/4/8-bit and MxN MAC extrapolation | Native 1/2/4/8-bit acquisition grids are preserved and converted only at the inference boundary; generalized array extrapolation remains pending | partial |
| R-2.1.1-D | Linearity, R² and nonlinear error | Existing fitting summary and plots | verified |
| R-2.1.2-A | Import array-MAC CSV with custom mapping | `mac_error_model.py`, GBK/UTF-8 and comma/semicolon tests | verified |
| R-2.1.2-B | MAC linearity/nonlinear model and persistence | Grouped linear/polynomial fits, metrics, integrity hash | verified |
| R-2.2-A | Four DNN, BNN and LLM | Registry contains usable LeNet-5, formally trained ResNet-20 and formally trained BNN; additional DNN/SNN/LLM types remain pending | partial |
| R-2.2-B | Four image datasets and local paths | MNIST and local CIFAR-10 verified | partial |
| R-2.2-C | Accuracy extrapolation model | Exact LUT inference and bounded ResNet differential evaluation exist; calibrated fast estimator pending | partial |
| R-2.2-D | Error-aware secondary training | LeNet-5 EAT/checkpoint/control experiment plus real cancellable GUI training workflow | verified |
| R-2.3-A | Data import and preview GUI | Real cell-error and array-MAC preview/validation workflows with GUI smoke evidence | verified |
| R-2.3-B | Array/bit configuration GUI | Array rows, columns and 1/2/4/8-bit controls drive real preview and fit filtering | verified |
| R-2.3-C | Model/dataset selection GUI | Live registry catalog and resource readiness implemented | partial |
| R-2.3-D | Training and results GUI | Real LeNet EAT and official NeuroSim PPA run in workers with progress, results, logs and retained artifacts | verified |
| R-2.3-E | Single-sample real-time inference GUI | Background GUI workflow runs real LeNet float/ideal/device inference and retains prediction/screenshot evidence | verified |
| R-2.4-A | CSV/XLSX export with generated names | GUI CSV/XLSX export works; naming refinement pending | partial |
| R-2.4-B | Remember last configuration | GUI remembers cell and array source/import/mapping/fit settings through QSettings | verified |
| R-2.4-C | View/clear/export logs | MAC GUI supports view, clear and UTF-8 TXT export | verified |
| R-2.4-D | Validation and robust error dialogs | Core exceptions exist; GUI handling pending | partial |
| R-3.1-A | Modular Python 3.8+ | Modular on Python 3.11; lower-version matrix unverified | partial |
| R-3.1-B | Windows high-DPI responsive GUI | PySide6 responsive MAC workflow and CJK font verified | partial |
| R-3.1-C | Efficient 100k-row data processing | 100k rows validated in 0.227 s on recorded PC | verified |
| R-3.2-A | Four-decimal linearity accuracy | Strict linear and known quadratic datasets match independent closed-form metrics within 1e-12 | verified |
| R-3.2-B | NeuroSim-referenced engineering model | Official C++ path verified for LeNet-5 and 22 ResNet-20 MAC layers; GUI PPA execution retains JSON, log, traces and screenshot | verified |
| R-3.2-C | <=1 s single sample, <=10 s batch | Cached real CPU inference measures about 10 ms per hot sample and 61 ms per hot 10-sample batch; cold preparation is reported separately | verified |
| R-3.3-A | Windows 10/11 CPU-only | current Windows run verified; clean-machine matrix pending | partial |
| R-3.3-B | CSV encoding/delimiter/missing policies | Both array-MAC and cell branches cover UTF-8/GBK, comma/semicolon and missing-value policies | verified |
| R-3.3-C | One-click dependency setup | `scripts/setup_environment.ps1` installs and checks the Python environment | verified |
| R-4.1-A | Portable Windows package | Runtime paths are configurable and the packaging boundary is machine-checked; executable build remains pending | partial |
| R-4.1-B | Installer/uninstaller/shortcut | pending | blocked |
| R-4.2-A | Complete organized source | active | partial |
| R-4.2-B | ≥30% comments and PEP 8 | final audit pending | implemented/unverified |
| R-4.2-C | Version-control history | workspace has no Git metadata; do not fabricate history | blocked |
| R-4.3-A | ResNet/MobileNet/SNN/BNN resources | pending; redistribution licenses must be checked | blocked |
| R-4.3-B | Bundled CIFAR-10/MNIST | local MNIST available; redistribution/package pending | partial |
| R-4.3-C | Cell and MAC example data | cell data exists; redistributable MAC sample pending | partial |
| R-4.4-A | User manual with real screenshots | `docs/USER_MANUAL.md` links verified GUI screenshots and executable workflows | verified |
| R-4.4-B | Technical document | `docs/TECHNICAL_MANUAL.md` documents architecture, numerical boundaries and artifacts | verified |
| R-4.4-C | Environment manual | `docs/ENVIRONMENT_SETUP.md` covers the verified Windows/Python setup and portable paths | verified |
| R-4.4-D | Test report | `docs/TEST_REPORT.md` links retained evidence and states verified and unverified scenarios | verified |
| R-5.1 | Complete workflows and working controls | core LeNet flow verified; full platform pending | partial |
| R-5.2 | Linearity ≤1% theory error and reasonable inference | Cell regression plus exact linear and independently computed nonlinear MAC theory cases are verified | verified |
| R-5.3-A | Startup/response/import/inference timing | 100k import passes; other timing scenarios pending | partial |
| R-5.3-B | Two-hour stability | final packaged build required | blocked |
| R-5.4 | Compatibility and robustness matrix | partial automated coverage | partial |
| R-5.5 | Complete delivery set | pending final build | blocked |
| R-6.1 | 30-calendar-day schedule | project/contract management | contract item |
| R-6.2 | 12-month support and response SLA | service obligation | contract item |
| R-7.1 | Originality and intellectual property | `THIRD_PARTY_NOTICES.md` and exclusion manifest identify restricted third-party assets; final legal review remains a contract activity | contract item |

## Licensing constraints

NeuroSim is CC BY-NC 4.0. Its copyright cannot be transferred as project-owned
source, and commercial redistribution requires separate legal review. ImageNet and
some checkpoints also cannot be assumed redistributable. The platform will provide
resource import, hashes and notices instead of silently bundling restricted assets.
