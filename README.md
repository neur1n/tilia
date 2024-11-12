# $\text{LIME}^2$

**L**IME with **IM**proved **E**xplanations


## 1 Prerequisites
The following prerequisites may not be strictly necessary, but they are the
versions that the code was tested with.

- Python >= 3.10
- PyTorch >= 2.5.1+cu124 (optional for image explanations)


## 2 Getting Started
The following instructions give a quick overview of what the code does.

### 2.1 Tabular Data
In the root directory of $\text{LIME}^2$:

```bash
cd script
python3 collect_tabular.py  # Collect explanations and related statistics from tabular data
python3 plot_tabular.py     # Visualiing collected statistics
```

### 2.2 Image Data (TODO)
In the root directory of $\text{LIME}^2$:

```bash
cd script
python3 collect_image.py  # Collect explanations and related statistics from image data
python3 plot_image.py     # Visualiing collected statistics
```


## 3 Usage
### 3.1 Data Collection
In the root directory of $\text{LIME}^2$, running the following command

```bash
cd script
python3 collect_tabular.py --help
```

will show the following help message (it may vary if this document is not up to date):

```bash
usage: collect_tabular.py [-h] [-r REGRESSOR] [-t TIMESTAMP]

options:
  -h, --help            show this help message and exit
  -r REGRESSOR, --regressor REGRESSOR
                        Regressor, linear or tree
  -t TIMESTAMP, --timestamp TIMESTAMP
                        Timestamp
```

where `--regressor` is the regressor type and only accepts `linear` or `tree`,
and `--timestamp` is the timestamp used to create the directory to store the
collected explanations and related statistics. If `--timestamp` is not
provided, the current date will be used, e.g., `20240101`.


### 3.2 Data Visualization
In the root directory of $\text{LIME}^2$, running the following command

```bash
cd script
python3 plot_tabular.py --help
```

will show the following help message (it may vary if this document is not up to date):

```bash
usage: plot_tabular.py [-h] [-d DATA] [-r REGRESSOR] [-t TIMESTAMP]

options:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Data used for plotting
  -r REGRESSOR, --regressor REGRESSOR
                        Regressor, linear or tree
  -t TIMESTAMP, --timestamp TIMESTAMP
                        Timestamp
```

where `--regressor` and `--timestamp` are the same as in the data collection,
and `--data` is the data used for visualizing the stability of each feature
importance which only accepts `mode`, `sum` (default), or `binned_sum`.
