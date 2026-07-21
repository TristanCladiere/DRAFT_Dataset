import torch
import os
import argparse
import time
import datetime
import logging

from model import FCLayer
from loss import CustomCELoss
from evaluator import RCFEvaluator, create_results_table
from torch.utils.tensorboard import SummaryWriter
from dataset import create_k_cross_dataset
from training import train_val_one_epoch, EarlyStopper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--data_folder', type=str, default='Data')
    parser.add_argument('--version', type=str, default='0.6B', choices=['0.6B', '4B', '8B'])
    parser.add_argument('--es', type=int, default=500)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--k_cross', type=str, default='1-5', help="first value is to choose the testing "
                                                                   "fold, second is to set the number of folds")

    _args = parser.parse_args()
    return _args

def main(args):
    start_time = time.time()

    # Instantiate output and tensorboard directories, and logger
    k_idx, k = args.k_cross.split("-")
    out_dir = (
        f"{args.out_dir}/{args.version}/ES-{args.es}_LR-{args.lr}/{k}_cross_val/{k_idx}"
    )

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    time_str = time.strftime("%d-%m-%Y-%H-%M-%S")
    console_log_file = f"{out_dir}/{time_str}"
    log_dir = out_dir.split("/")[1:]
    log_dir = "tensorboard_logs/" + "/".join(log_dir)

    writer = dict()
    for phase in ["Train", "Val"]:
        try:
            os.makedirs(os.path.join(log_dir, phase))
        except FileExistsError:
            pass
        writer[phase] = SummaryWriter(log_dir=f"{log_dir}/{phase}")

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=console_log_file, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    device = torch.device("cuda")

    # Create Train/Val/Test sets
    test_data, pairs, labels, sas = create_k_cross_dataset(args.data_folder, args.version, int(k_idx) - 1, int(k))

    # Create the learnable model (Single FC layer)
    model = FCLayer(args.version)
    model.to(device)

    # Resume training if necessary
    epoch = 0
    snapshot_path = f'{out_dir}/snapshot.pth'
    best_path = f'{out_dir}/best_model.pth'
    if os.path.exists(snapshot_path):
        loc = f"cuda"
        snapshot = torch.load(snapshot_path, map_location=loc)
        model.load_state_dict(snapshot["model"])
        epoch = snapshot["epoch"]
        print(f"Resuming training from snapshot at Epoch {epoch}")

    # Instantiate loss
    class_weights = dict()
    for part_name in ["Train", "Val"]:
        pairs[part_name] = torch.stack(pairs[part_name]).to(torch.float32).to(device)
        labels[part_name] = torch.stack(labels[part_name]).to(torch.int64).to(device)

        # To take into account the class imbalance between positive and negative pairs
        num_pos = (labels[part_name] == 1).sum().item()
        num_neg = (labels[part_name] == 0).sum().item()
        total = len(labels[part_name])
        weight_pos = total / (2.0 * num_pos)
        weight_neg = total / (2.0 * num_neg)
        class_weights[part_name] = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(device)
    criterion = CustomCELoss(weights=class_weights, labels=labels)

    # Instantiate optimizer and early stopping
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    early_stopper = EarlyStopper(patience=args.es, best_path=best_path, snapshot_path=snapshot_path)

    # Update the optimizer if training resumes.
    if os.path.exists(snapshot_path):
        optimizer.load_state_dict(snapshot["optimizer"])

    # Run training until early stopper triggers
    while not early_stopper.early_stop:
        epoch = train_val_one_epoch(
            model, pairs, optimizer, early_stopper, criterion, epoch, logger, writer
        )
    logger.info("")
    logger.info(f"Early stopping triggered")

    for w in writer.values():
        w.close()

    train_time = time.time() - start_time
    train_time_str = str(datetime.timedelta(seconds=int(train_time)))
    logger.info("")
    logger.info(f"Training completed. Total time {train_time_str}")
    logger.info("")

    # Start testing the best model
    logger.info("Starting best model testing")
    logger.info("")
    start_test = time.time()
    evaluator = RCFEvaluator(ranking_data=test_data,
                             classif_data=labels["Test"],
                             mrr_at_k=[3],
                             ndcg_at_k=[3],
                             hr_at_k=[3],
                             precision_recall_at_k=[3],
                             map_at_k=[3])

    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    scores, preds = model.compute_scores_and_preds(pairs["Test"].to(device))
    results, txt_job_rank, txt_job_fair = evaluator(scores, preds, save_dir_path=out_dir)
    logger.info(txt_job_rank)
    logger.info(txt_job_fair)
    logger.info("")
    logger.info(create_results_table(results))
    test_time = time.time() - start_test
    test_time_str = str(datetime.timedelta(seconds=int(test_time)))
    logger.info("")
    logger.info(f"Test completed. Total time {test_time_str}")


if __name__ == "__main__":
    main(parse_args())
