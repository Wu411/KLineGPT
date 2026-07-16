  # Data Directory

Place your OHLCV market data here. Raw data is **not** committed to the repository
(see `.gitignore`).

## Expected structure

```
data/
├── output_kline/
│   ├── csi300.csv        # per-stock daily OHLCV
│   ├── csi800.csv
│   └── sp500.csv
└── output_index_data/
    ├── csi300_index.csv  # benchmark index OHLCV
    ├── csi800_index.csv
    └── sp500_index.csv
```

## Stock K-line file format (`output_kline/*.csv`)

Long format, one row per (stock, day):

| column   | type   | description               |
|----------|--------|---------------------------|
| `kdcode` | str    | stock identifier          |
| `dt`     | date   | trading date (YYYY-MM-DD) |
| `open`   | float  | open price                |
| `high`   | float  | high price                |
| `low`    | float  | low price                 |
| `close`  | float  | close price               |
| `volume` | float  | trading volume            |

## Index file format (`output_index_data/*_index.csv`)

Daily benchmark index series with a `dt` (or `date`) column and at least a
`close` column (`open/high/low/volume` recommended for market-context encoding).

> All data must be **point-in-time** to avoid look-ahead / survivorship bias.
