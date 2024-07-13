# neural-probabilistic-model-bengioatall

## References
- https://medium.com/@dahami/a-neural-probabilistic-language-model-breaking-down-bengios-approach-4bf793a84426

my_pytorch_project/
├── data/
│   ├── raw/                # Raw data downloaded from the source
│   ├── processed/          # Processed data ready for training
│   ├── dataset.py       # Custom Dataset class
│   └── data_preparation.py # Scripts to download/generate data
├── models/
│   ├── __init__.py         # Model initialization file
│   ├── model.py            # Model architecture
│   └── utils.py            # Model utilities
├── notebooks/
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   └── experiments.ipynb   # Experiment tracking
├── scripts/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── predict.py          # Prediction script
├── tests/
│   ├── test_data.py        # Tests for data processing
│   ├── test_model.py       # Tests for model
│   └── test_training.py    # Tests for training
├── configs/
│   ├── config.yaml         # Configuration file
│   └── hyperparameters.yaml# Hyperparameter configuration
├── logs/
│   └── training_logs/      # Training logs
├── checkpoints/
│   └── model_checkpoints/  # Saved model checkpoints
├── requirements.txt        # Required packages
├── README.md               # Project description and instructions
└── setup.py                # Installation script



export PYTHONPATH=$PYTHONPATH:$(pwd)
