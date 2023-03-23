
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=64, help="_")
    parser.add_argument("--img_per_place", type=int, default=4, help="_")
    parser.add_argument("--min_img_per_place", type=int, default=4, help="_")
    parser.add_argument("--descriptors_dim", type=int, default=512, help="_")
    parser.add_argument("--max_epochs", type=int, default=20, help="_")
    parser.add_argument("--num_workers", type=int, default=16, help="_")

    parser.add_argument("--train_path", type=str, required=True, help="path to train set")
    parser.add_argument("--val_path", type=str, required=True, help="path to val set")
    parser.add_argument("--test_path", type=str, required=True, help="path to test set")
    args = parser.parse_args()
    return args

