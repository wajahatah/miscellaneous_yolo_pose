import os
from collections import defaultdict

def count_instances(labels_dir):
    class_counts = defaultdict(int)
    total_instances = 0

    for file in os.listdir(labels_dir):
        if not file.endswith(".txt"):
            continue

        with open(os.path.join(labels_dir, file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue

                class_id = parts[0]   # YOLO class is always first element
                class_counts[class_id] += 1
                total_instances += 1

    return class_counts, total_instances


def main(dataset_path):
    train_dir = os.path.join(dataset_path, "labels/train")
    val_dir = os.path.join(dataset_path, "labels/val")

    # TRAIN
    train_counts, train_total = count_instances(train_dir)
    print("\n=== TRAIN ===")
    print("Class counts:", dict(train_counts))
    print("Total:", train_total)

    # VAL
    val_counts, val_total = count_instances(val_dir)
    print("\n=== VAL ===")
    print("Class counts:", dict(val_counts))
    print("Total:", val_total)

    # OVERALL
    overall = defaultdict(int)
    for c, v in train_counts.items():
        overall[c] += v
    for c, v in val_counts.items():
        overall[c] += v

    print("\n=== OVERALL ===")
    print("Class counts:", dict(overall))
    print("Total:", train_total + val_total)


if __name__ == "__main__":
    import os
    main("C:/wajahat/hand_in_pocket/dataset/images_bb/training1")
