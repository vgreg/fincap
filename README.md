# #fincap - code description

This code contains the analysis for our contribution to the [Finance Crowd Analysis Project](https://fincap.academy/). For more information, see the [resulting paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3961574).

Team id GRFN, [Vincent Grégoire](https://www.vincentgregoire.com/) (HEC Montreal) and [Charles Martineau](https://www.charlesmartineau.com/) (University of Toronto). Our "paper" was one of the five best rated papers that were shared with all the teams in the last phase of the project.

## Requirements

Our code is implemented in Python using standard scientific packages.
We recommend using the [Anaconda](https://www.anaconda.com/) distribution, which will install almost all required packages by default (pandas, numpy, matplotlib, etc.) The only other dependancy is the pyarrow package that we use to read and write files in the parquet format for storing intermediary results. The code has been tested on Python 3.8.5 with pandas version 1.1.3.

## Files

Our code assumes the following directory structure:
```
.
+-- Code/
|   +-- Analysis.ipynb
|   +-- ComputeMeasures.py
+-- Data/
|   +-- DailyMeasures_v2.parquet
|   +-- raw-data-from-deutsche-boerse-for-fincap/
|       +-- 2002-01.csv.gz
|       +-- 2002-02.csv.gz
|       +-- 2002-03.csv.gz
|       +-- ...
+-- Paper/
|   +-- Figures/
|   +-- Tables/
|   +-- Text_Python/
+-- README.md
```

We have three code files, which should be executed in the following order:

1. `ComputeMeasures.py` : This code computes daily measures from the raw data files.
1. `ExtractTradingDays.py` : This code computes actual expiration dates from trading days.
2. `Analysis.ipynb` : This code performs the statistical analysis and exports results.

## ComputeMeasures.py

This Python script processes all monthly csv files in `raw-data-from-deutsche-boerse-for-fincap` and computes daily measures of interest for each contract.

Most of the file contains functions that are called at the end of the file. The main execution is in the code block
under `if __name__ == '__main__':`. That code block reads the list of `csv.gz` files to process and then calls `process_monthly_file(fn)` for each monthly file in parallel using all available CPU cores. The execution time is less than 6 minutes on a 10-core Intel Core i9 cpu with 128GB of RAM running Ubuntu 20.04.

The sole output is the file `DailyMeasures_v3.parquet`. This is a parquet file, a format for tabular data like CSV but with binary encoding for more efficient reading and writing. It contains the following columns:

- `DATE`: Observation date. (index column)
- `EXPIRATION`: Expiration year/month  of the contract.
- `ABS_VR_30T5T`: Absolute difference between 1 and the variance ratio computed from 30-minute returns over 5-minute returns.
- `SHR_CLIENT_VOL`: Fraction of total volume made up by client trades. 
- `TOTAL_QTY`: Total volume.
- `CLIENTS_QTY`: Client volume.
- `REAL_SPREAD_5T`: Volume-weighted average realized spread.
- `REAL_SPREAD_5T_QTY`, Total volume used for the calculation of `REAL_SPREAD_5T`.
- `CLIENTS_REAL_SPREAD_5T`: Volume-weighted average realized spread for client trades.
- `CLIENTS_REAL_SPREAD_5T_QTY`: Total volume used for the calculation of `CLIENTS_REAL_SPREAD_5T`.
- `FRAC_CLIENTS_MKT`: Fraction of client trades that are market orders or marketable limit orders.
- `CLIENTS_MKT_QTY`: Total volume used in the numerator of `FRAC_CLIENTS_MKT`.
- `GTR`: Dollar volume-weighted average relative gross trading revenue for client trades.
- `GTR_VALUE`: Total dollar volume used for the calculation of `GTR`.

## ExtractTradingDays.py

This Python script processes all monthly csv files in `raw-data-from-deutsche-boerse-for-fincap` and computes the actual expiration date from
trading days.

Most of the file contains functions that are called at the end of the file. The main execution is in the code block
under `if __name__ == '__main__':`. That code block reads the list of `csv.gz` files to process and then calls `process_monthly_file(fn)` for each monthly file in parallel using all available CPU cores. The execution time is less than 6 minutes on a 10-core Intel Core i9 cpu with 128GB of RAM running Ubuntu 20.04.

The sole output is the file `Expirations.parquet`. This is a parquet file, a format for tabular data like CSV but with binary encoding for more efficient reading and writing. It contains the following columns:

- `EXPIRATION`: Expiration year/month of the contract.
- `EXPIRATION_DATE`: Expiration date of the contract.

## Analysis.ipynb

This Jupyter notebooks reads in `DailyMeasures_v3.parquet` and `Expirations.parquet`. The code computes monthly measures from the daily measures and performs the statstical analysis.
The code outputs the file `Timeseries_monthly.pdf` which plots the time series of all monthly measures, the file `RegResult.tex` that contains a table with the estimation results, and `.tex` files under `Text_Python/` that contains the textual description of the statistical results to be included in the paper.
