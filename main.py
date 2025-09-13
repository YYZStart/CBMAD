import argparse
import os
from evaluate_and_store import generate_data_for_ad, model_inferences_ad
from model.CBMAD_model import CBMAD


def main():
    parser = argparse.ArgumentParser(description="Run CBMAD anomaly detection")

    parser.add_argument(
        "--dataset", type=str, default="SMD",
        help="Dataset name (default: SMD)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=100,
        help="Sequence length (default: 100)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Number of epochs (default: 40)"
    )
    parser.add_argument(
        "--nb_feature", type=int, default=38,
        help="Feature dimension (default: 38 for SMD)"
    )

    parser.add_argument(
        "--state_size", type=int, default=16,
        help="State size for the model (default: 16)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2,
        help="Number of layers (default: 2)"
    )

    parser.add_argument('--rms_norm', dest='rms_norm', action='store_true')
    parser.add_argument('--no-rms_norm', dest='rms_norm', action='store_false')
    parser.set_defaults(rms_norm=True)

    parser.add_argument(
        "--alpha", type=float, default=0.7,
        help="Alpha parameter (default: 0.7)"
    )

    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--action", type=str, default="train_and_infer",
        choices=["train_and_infer", "infer"],
        help="Action to perform: train_and_infer or infer (default: train_and_infer)"
    )

    args = parser.parse_args()

    model_name = "CBMAD"
    ModelClass = CBMAD


    X_train, X_test, X_test_label = generate_data_for_ad(args.dataset, "test") # test.csv depend on the file name

    # checkpoint path
    model_dir = os.path.join("./checkpoints", model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f"{model_name}_{args.dataset}.pt")


    model_kwargs = {
        "nb_feature": args.nb_feature,
        "state_size": args.state_size,
        "num_layers": args.num_layers,
        "rms_norm": args.rms_norm,
        "alpha": args.alpha,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "device": args.device,
    }

    if args.action == "infer" and os.path.exists(model_file):
        print(f"Loading model for inference: {model_file}")
        clf = ModelClass(**model_kwargs)
        clf.load_pt_model(model_file)

    elif args.action == "train_and_infer":
        print(f"Training model for dataset {args.dataset} with config:\n{model_kwargs}")
        clf = ModelClass(**model_kwargs)
        clf.fit(X_train)

        print(f"Saving trained model to {model_file}")
        clf.save_pt_model(model_file)

    else:
        print(f"Model file not found for inference: {model_file}")
        return

    model_inferences_ad(X_test, X_test_label,
                        file_key=f"{args.dataset}",
                        clf=clf, model_name=model_name)


if __name__ == "__main__":
    main()
