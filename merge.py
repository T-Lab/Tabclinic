import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================================================
# 计算评估指标
# =========================================================
def compute_metrics(y_true, y_pred):

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }


# =========================================================
# 读取 consensus prediction matrix
# =========================================================
def load_matrix(path):

    data = np.load(path)

    return data["predictions"], data["confidence"]


# =========================================================
# merge 多个 block
# =========================================================
def merge_blocks(
        block_paths,
        dirty_blocks,
        clean_blocks,
        outdir
):

    # -----------------------------
    # 1 读取所有 block 预测
    # -----------------------------
    preds_list = []
    conf_list = []

    for p in block_paths:

        print("Loading prediction:", p)

        pred, conf = load_matrix(p)

        preds_list.append(pred)
        conf_list.append(conf)

    # 合并预测矩阵
    final_pred = np.vstack(preds_list)
    final_conf = np.vstack(conf_list)

    print("Merged prediction shape:", final_pred.shape)

    # -----------------------------
    # 2 读取 dirty / clean
    # -----------------------------
    dirty_list = []
    clean_list = []

    for p in dirty_blocks:
        print("Loading dirty:", p)
        dirty_list.append(pd.read_csv(p, dtype=str))

    for p in clean_blocks:
        print("Loading clean:", p)
        clean_list.append(pd.read_csv(p, dtype=str))

    dirty = pd.concat(dirty_list, ignore_index=True)
    clean = pd.concat(clean_list, ignore_index=True)

    print("Merged dirty shape:", dirty.shape)

    # -----------------------------
    # 3 安全检查
    # -----------------------------
    assert dirty.shape == clean.shape, "Dirty and clean shape mismatch"
    assert dirty.shape == final_pred.shape, "Prediction and data row mismatch"

    num_rows, num_cols = final_pred.shape

    os.makedirs(outdir, exist_ok=True)

    # =====================================================
    # 4 保存 consensus matrix
    # =====================================================
    np.savez_compressed(
        os.path.join(outdir, "consensus_predictions_matrix.npz"),
        predictions=final_pred,
        confidence=final_conf
    )

    print("Saved consensus_predictions_matrix.npz")

    # =====================================================
    # 5 生成 predictions.csv
    # =====================================================
    preds_df = pd.DataFrame(final_pred, columns=dirty.columns)

    preds_df.to_csv(
        os.path.join(outdir, "predictions.csv"),
        index=False
    )

    print("Saved predictions.csv")

    # =====================================================
    # 6 计算 ground truth
    # =====================================================
    gt = (dirty != clean).astype(int)

    # =====================================================
    # 7 生成 ensemble_voting_details.csv
    # =====================================================
    rows = np.repeat(np.arange(num_rows), num_cols)
    cols = np.tile(np.arange(num_cols), num_rows)

    df = pd.DataFrame({
        "row_id": rows,
        "column": [dirty.columns[c] for c in cols],
        "dirty_value": dirty.values.flatten(),
        "clean_value": clean.values.flatten(),
        "ground_truth": gt.values.flatten(),
        "ensemble_prediction": final_pred.flatten(),
        "confidence": final_conf.flatten()
    })

    df.to_csv(
        os.path.join(outdir, "ensemble_voting_details.csv"),
        index=False
    )

    print("Saved ensemble_voting_details.csv")

    # =====================================================
    # 8 计算 metrics
    # =====================================================
    overall = compute_metrics(
        gt.values.flatten(),
        final_pred.flatten()
    )

    per_column = {}

    for i, col in enumerate(dirty.columns):

        per_column[col] = compute_metrics(
            gt.values[:, i],
            final_pred[:, i]
        )

    metrics = {
        "overall": overall,
        "per_column": per_column
    }

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics.json")

    print("\nMerge complete")


# =========================================================
# main
# =========================================================
if __name__ == "__main__":

    # prediction blocks
    block_paths = [

        # 前333
        "/Users/zihaoxie/Desktop/comp9991/ForestED-master/data/hospital_output/consensus_predictions_matrix.npz",

        # 中333
        "/Users/zihaoxie/Desktop/comp9991/ForestED-master/data/hospital_output2/consensus_predictions_matrix.npz"
    ]

    # dirty blocks
    dirty_blocks = [

        "/Users/zihaoxie/Desktop/comp9991/ForestED-master/data/split_output_dirty/part1.csv",

        "/Users/zihaoxie/Desktop/comp9991/ForestED-master/data/split_output_dirty/part2.csv"
    ]

    # clean blocks
    clean_blocks = [

        "/Users/zihaoxie/Desktop/comp9991/ForestED-master/data/split_output_clean/part1.csv",

        "/Users/zihaoxie/Desktop/comp9991/ForestED-master/data/split_output_clean/part2.csv"
    ]

    outdir = "/Users/zihaoxie/Desktop/comp9991/ForestED-master/data/merged_output"

    merge_blocks(
        block_paths,
        dirty_blocks,
        clean_blocks,
        outdir
    )