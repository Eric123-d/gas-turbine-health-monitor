# Data source

`GTPP-DATASET.csv` is the public **SMART-GTPP Dataset**:

- Asset: utility-scale GE Frame 9E industrial gas turbine
- Measurements: GE Mark VIe control system and AVEVA PI historian
- Rows: 7,893
- Variables: eight inputs and electrical power output `EP` in MW
- Sampling interval stated by the companion paper: 4 minutes
- Dataset DOI: https://doi.org/10.17632/6sk3mhm7hb.1
- Companion paper: https://doi.org/10.21070/jeeeu.v10i1.1738
- Data license: CC BY 4.0

Important limitation: the public CSV does not contain absolute timestamps,
health labels, fault labels, maintenance records, or unit identifiers.  This
project therefore calls it **assumed-normal operating data**, never
label-confirmed healthy data.
