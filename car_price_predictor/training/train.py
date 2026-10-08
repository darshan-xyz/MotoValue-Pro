"""
CLI entrypoint alias allowing 'python -m car_price_predictor.training.train'.
"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
