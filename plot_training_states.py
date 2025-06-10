import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_last_ckpt(output_dir: Path):
    cands = [ckpt_path for ckpt_path in output_dir.iterdir() if 'checkpoint-' in str(ckpt_path)]
    steps = [int(str(ckpt_path).split('-')[-1]) for ckpt_path in cands]
    return cands[np.argmax(steps)]


OUTPUT_DIR = Path('.output_epoch=10')
for dataset_name in ['SNLI', 'MNLI', 'ANLI']:

    output_dir = OUTPUT_DIR / dataset_name
    last_ckpt_path = find_last_ckpt(output_dir)
    print('last checkpoint path:', last_ckpt_path)

    trainer_state_path = last_ckpt_path / 'trainer_state.json'
    trainer_state = json.loads(trainer_state_path.read_text())

    epoch = []
    training_loss = []
    eval_loss = []
    eval_accuracy = []
    eval_precision = []
    eval_recall = []
    eval_f1 = []

    for log in trainer_state['log_history']:
        if 'loss' in log:
            epoch.append(log['epoch'])
            training_loss.append(log['loss'])
        else:
            eval_loss.append(log['eval_loss'])
            eval_accuracy.append(log['eval_accuracy'])
            eval_precision.append(log['eval_precision'])
            eval_recall.append(log['eval_recall'])
            eval_f1.append(log['eval_f1'])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axs[0].plot(epoch, training_loss, label='Training Loss', marker='x')
    axs[0].plot(epoch, eval_loss, label='Evaluation Loss', marker='x')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(epoch)
    axs[0].legend()
    axs[1].plot(epoch, eval_accuracy, label='Evaluation Accuracy', marker='x')
    axs[1].plot(epoch, eval_precision, label='Evaluation Precision', marker='x')
    axs[1].plot(epoch, eval_recall, label='Evaluation Recall', marker='x')
    axs[1].plot(epoch, eval_f1, label='Evaluation F1', marker='x')
    # axs[1].set_ylim(0.45, 0.95)
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(epoch)
    axs[1].legend()

    best_idx = np.argmax(eval_accuracy)
    best_epoch = epoch[best_idx]
    best_accuracy = eval_accuracy[best_idx]
    axs[1].axvline(x=best_epoch, linestyle='--', color='gray', alpha=0.6)
    axs[1].axhline(y=best_accuracy, linestyle='--', color='gray', alpha=0.6)
    axs[1].text(
              axs[1].get_xlim()[1] + 1
            , best_accuracy
            , f'Best Acc.\n {best_accuracy:.4f}'
            , ha='center'
            , va='center'
            , fontsize=10
            , color='black'
            , rotation=0
        )

    fig_path = Path('train_state')
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_path = fig_path / '{}.png'.format(dataset_name)
    fig.savefig(str(fig_path))
    plt.close()  # HACK
