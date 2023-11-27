import argparse
from funkcje import train_and_export_model, predict_unknown

# python src/app.py train --X 12 --Y 23.65
# python src/app.py predict --X 2.78

def main():
    parser = argparse.ArgumentParser(description="Linear Regression Model Tester")
    parser.add_argument("mode", choices=["train", "predict"], help="Choose mode: 'train' or 'predict'")
    parser.add_argument("--X", type=float, help="Input value for prediction")
    parser.add_argument("--Y", type=float, help="Output value for training")
    parser.add_argument("--csv_path", default="data/10_points.csv", help="Path to the CSV file")

    args = parser.parse_args()

    if args.mode == "train":
        if args.X is None or args.Y is None:
            print("Please provide both X and Y values for training.")
            return
        train_and_export_model(args.X, args.Y, args.csv_path)
    elif args.mode == "predict":
        if args.X is None:
            print("Please provide an X value for prediction.")
            return
        prediction = predict_unknown(args.X)
        print(f"The predicted Y value for X = {args.X} is {prediction}")

if __name__ == "__main__":
    main()
