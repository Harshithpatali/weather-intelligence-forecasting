"""Train and evaluate fixed-lead India weather calibration models."""
import argparse
import json
from ml.india_training import train_india_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/india_forecast_training.parquet")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--minimum-locations", type=int, default=30)
    parser.add_argument("--minimum-days", type=int, default=300)
    args = parser.parse_args()
    report = train_india_dataset(
        args.data, args.output_dir,
        minimum_locations=args.minimum_locations,
        minimum_days=args.minimum_days,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
