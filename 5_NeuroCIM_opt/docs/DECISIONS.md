# Engineering decisions

## D-001: Preserve one source tree

All changes are made in place. Generated artifacts use run directories under
`outputs`; source copies with version suffixes are prohibited.

## D-002: Separate numerical error from NeuroSim PPA

The measured LUT/MAC model drives inference and training. Official NeuroSim drives
area, latency and energy estimation. This matches the implemented upstream API and
prevents unsupported claims that NeuroSim reads the measured LUT.

## D-003: Use explicit grouped MAC models

Array-MAC observations are fitted independently by rows, columns, bit width and
experiment group. Prediction outside the measured ideal-MAC range is rejected by
default, because silent extrapolation has no experimental support.

## D-004: Do not encode contractual promises as software completion

Support SLAs, development duration and copyright assignment are tracked separately.
Third-party licenses and dataset redistribution rights remain visible in delivery.
