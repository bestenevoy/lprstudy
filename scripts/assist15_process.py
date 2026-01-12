import csv
import random
from collections import defaultdict
from pathlib import Path


def load_sessions(csv_path):
    sessions = defaultdict(list)
    sequence_ids = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row["user_id"]
            log_id = int(row["log_id"])
            sequence_id = int(row["sequence_id"])
            correct_raw = row["correct"].strip()
            try:
                correct = int(correct_raw)
            except ValueError:
                correct = 1 if float(correct_raw) >= 0.5 else 0
            sequence_ids.add(sequence_id)
            sessions[user_id].append((log_id, sequence_id, correct))

    mapping = {seq_id: i + 1 for i, seq_id in enumerate(sorted(sequence_ids))}
    mapped_sessions = {}
    for user_id, logs in sessions.items():
        logs.sort(key=lambda x: x[0])
        mapped_sessions[user_id] = [
            (mapping[sequence_id], correct) for _, sequence_id, correct in logs
        ]

    return mapped_sessions, mapping


def split_users(users, fold_count, seed):
    rng = random.Random(seed)
    users = list(users)
    rng.shuffle(users)
    folds = [[] for _ in range(fold_count)]
    for i, user_id in enumerate(users):
        folds[i % fold_count].append(user_id)
    return folds


def split_train_val_test(users, train_ratio, val_ratio):
    n_total = len(users)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    return (
        users[:n_train],
        users[n_train : n_train + n_val],
        users[n_train + n_val : n_train + n_val + n_test],
    )


def write_split(output_dir, split_name, user_ids, sessions):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx, user_id in enumerate(user_ids):
        out_path = split_dir / f"{idx}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["problem_id", "correct"])
            for problem_id, correct in sessions[user_id]:
                writer.writerow([problem_id, correct])


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "ASSISTments2015"
    csv_path = raw_dir / "2015_100_skill_builders_main_problems.csv"
    output_base = raw_dir / "processed"

    sessions, mapping = load_sessions(csv_path)
    if len(mapping) != 100:
        raise ValueError(f"Expected 100 unique sequence_id values, got {len(mapping)}")

    folds = split_users(sessions.keys(), fold_count=5, seed=202401)
    for fold_index, fold_users in enumerate(folds, start=1):
        fold_users = list(fold_users)
        train_users, val_users, test_users = split_train_val_test(
            fold_users, train_ratio=0.7, val_ratio=0.2
        )
        fold_dir = output_base / str(fold_index)
        write_split(fold_dir, "train", train_users, sessions)
        write_split(fold_dir, "val", val_users, sessions)
        write_split(fold_dir, "test", test_users, sessions)

    print(f"Wrote processed data to {output_base.as_posix()}")


if __name__ == "__main__":
    main()
