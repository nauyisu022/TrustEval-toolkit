# LLM Truthfulness

## File Structure

```python
project_root/
│
├── src/
│
├── cache/
│
├── section/                
│   ├── truthfulness/           # source code for truthfulness
│   │   ├── __init__.py         # Makes truthfulness a package
│   │   ├── dataset_pool.py     # create dataset pool from existing dataset
│   │   ├── dynamic.py          # create dynamic dataset
│   │   ├── intermediate/       # intermediate results
│   │   ├── final/              # final datasets
│   │   ├── README.md
│
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
│
│
└── truthfulness_main.py    # Sample file for running truthfulness
```

## Usage

### Create Dataset Pool

```python
from section.truthfulness.dataset_pool import TruthfulnessDP
TruthfulnessDP.run()
```

### Create Dynamic Dataset

```python
from section.truthfulness.dynamic import TruthfulnessDynamic
TruthfulnessDynamic.run()
```

## Contact

If you have any questions or need further clarification, please contact [Haoran Wang](hwang219@hawk.iit.edu).
