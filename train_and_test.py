# train and test NLI model whose base-model is RoBERTa-large
# ref: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py

import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers.data import DataCollatorWithPadding


@dataclass
class NLIDatasetInfo:
    name: str
    hf_path: str
    pre_col: str
    hypo_col: str
    label_col: str
    labels: List[Any]
    train_splits: List[str]
    eval_splits: List[str]
    test_splits: List[str]


# see: https://huggingface.co/datasets/stanfordnlp/snli
SNLI_INFO = NLIDatasetInfo(
          name='SNLI'
        , hf_path='stanfordnlp/snli'
        , pre_col='premise'
        , hypo_col='hypothesis'
        , label_col='label'
        , labels=[
                0,  # entailment
                1,  # neutral
                2,  # contradiction
            ]
        , train_splits=['train']
        , eval_splits=['validation']
        , test_splits=['test']
    )

# see: https://huggingface.co/datasets/nyu-mll/multi_nli
MNLI_INFO = NLIDatasetInfo(
          name='MNLI'
        , hf_path='nyu-mll/multi_nli'
        , pre_col='premise'
        , hypo_col='hypothesis'
        , label_col='label'
        , labels=[
                0,  # entailment
                1,  # neutral
                2,  # contradiction
            ]
        , train_splits=['train']
        , eval_splits=['validation_matched', 'validation_mismatched']
        , test_splits=[]
    )

# see: https://huggingface.co/datasets/facebook/anli
ANLI_INFO = NLIDatasetInfo(
          name='ANLI'
        , hf_path='facebook/anli'
        , pre_col='premise'
        , hypo_col='hypothesis'
        , label_col='label'
        , labels=[
                0,  # entailment
                1,  # neutral
                2,  # contradiction
            ]
        , train_splits=['train_r1', 'train_r2', 'train_r3']
        , eval_splits=['dev_r1', 'dev_r2', 'dev_r3']
        , test_splits=['test_r1', 'test_r2', 'test_r3']
    )

DATASET_CACHE_DIR = '.dataset'
MODEL_CACHE_DIR = '.model'
OUTPUT_DIR = '.output'

CLASSLABEL_LIST = [
      'entailment'     # 0
    , 'neutral'        # 1
    , 'contradiction'  # 2
]


#######################################################################################################


def main(do_train: bool = False):

    tokenizer, model = load_pretrained_model('roberta-large', CLASSLABEL_LIST)

    if do_train:
        results_after_training = []
        for nli_dataset_info in [SNLI_INFO, MNLI_INFO, ANLI_INFO]:
            output_dir = os.path.join(OUTPUT_DIR, nli_dataset_info.name)
            training_args = TrainingArguments(
                      output_dir=output_dir
                    , overwrite_output_dir=True         # to overwrite the output directory
                    , do_train=True
                    , do_eval=True
                    , eval_strategy='epoch'             # to evaluate every epoch
                    , save_strategy='epoch'             # to save the model every epoch
                    , logging_strategy='epoch'          # to log every epoch
                    , learning_rate=1e-5                # equivalent to DocNLI
                    , weight_decay=1e-2                 # to regularize
                    , num_train_epochs=10               # equivalent to 2 * DocNLI
                    , per_device_train_batch_size=16
                    , gradient_accumulation_steps=2     # batch_size ~ this * per_device_train_epoch_batch_size
                    , per_device_eval_batch_size=16
                    , fp16=torch.cuda.is_available()    # to use mixed precision training
                    , load_best_model_at_end=True       # to select best model checkpoint(epoch) for next trainee
                    , metric_for_best_model='accuracy'  # metric to determine best model checkpoint
                )
            
            tokenizer, model = train(nli_dataset_info, tokenizer, model, training_args)
            results = test(nli_dataset_info, tokenizer, model)
            results_after_training.append(results)
        print(results_after_training)


    save_dir = os.path.join(OUTPUT_DIR + '_epoch=10', 'ANLI', 'save')
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)

    for nli_dataset_info in [SNLI_INFO, MNLI_INFO, ANLI_INFO]:
        test(nli_dataset_info, tokenizer, model)


#######################################################################################################


def load_pretrained_model(
      pretrained_model_name: str
    , classlabel_list: list[Any]
) \
    -> Tuple[PreTrainedTokenizer, PreTrainedModel]:

    label2id = { v: i for i, v in enumerate(classlabel_list) }
    id2label = { v: k for k, v in label2id.items() }
    config = AutoConfig.from_pretrained(
              pretrained_model_name_or_path=pretrained_model_name
            , num_labels=len(classlabel_list)
            , finetuning_task='text-classification'
            , cache_dir=MODEL_CACHE_DIR
            , revision='main'
        )
    tokenizer = AutoTokenizer.from_pretrained(
              pretrained_model_name_or_path=pretrained_model_name
            , cache_dir=MODEL_CACHE_DIR
            , revision='main'
            , use_fast_tokenizer=True
        )
    model = AutoModelForSequenceClassification.from_pretrained(
              pretrained_model_name_or_path=pretrained_model_name
            , config=config
            , cache_dir=MODEL_CACHE_DIR
            , revision='main'
        )
    model.config.label2id = label2id
    model.config.id2label = id2label
    return tokenizer, model


def train(
      nli_dataset_info: NLIDatasetInfo
    , tokenizer: PreTrainedTokenizer
    , model: PreTrainedModel
    , training_args: TrainingArguments
) \
    -> Tuple[PreTrainedTokenizer, PreTrainedModel]:

    if nli_dataset_info.train_splits == [] or nli_dataset_info.eval_splits == []:
        raise RuntimeError('{} does not have train and eval splits.'.format(nli_dataset_info.name))
    
    print('Preparing {} dataset...'.format(nli_dataset_info.name))
    dataset = load_dataset(nli_dataset_info.hf_path, cache_dir=DATASET_CACHE_DIR)
    dataset_train, dataset_eval, _ = split_dataset(nli_dataset_info, dataset)

    dataset_train = preprocess_dataset(nli_dataset_info, dataset_train, tokenizer)
    dataset_eval = preprocess_dataset(nli_dataset_info, dataset_eval, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
              model=model
            , args=training_args
            , train_dataset=dataset_train
            , eval_dataset=dataset_eval
            , compute_metrics=compute_metrics
            , processing_class=tokenizer
            , data_collator=data_collator
        )

    print('Start training on {}...'.format(nli_dataset_info.name))
    save_dir = os.path.join(training_args.output_dir, 'save')
    try:
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model(output_dir=save_dir)
        trainer.save_metrics(split='train', metrics=train_result.metrics)
        trainer.save_metrics(split='eval', metrics=train_result.metrics)

    except KeyboardInterrupt:
        # HACK: when you interrrpt the training, GPU may not be initialized properly
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise KeyboardInterrupt('Training interrupted by user.')
    print('Done training on {}.'.format(nli_dataset_info.name))
    return tokenizer, model


def test(
      nli_dataset_info: NLIDatasetInfo
    , tokenizer: PreTrainedTokenizer
    , model: PreTrainedModel
) \
    -> Tuple[Dict[str, Dict[str, Dict[str, float]]]]:

    print('Prepare dataset for {}...'.format(nli_dataset_info.name))
    dataset = load_dataset(nli_dataset_info.hf_path, cache_dir=DATASET_CACHE_DIR)
    _, dataset_eval, dataset_test = split_dataset(nli_dataset_info, dataset)

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
              model=model
            , compute_metrics=compute_metrics
            , processing_class=tokenizer
            , data_collator=data_collator
        )

    print('Running prediction on {}...'.format(nli_dataset_info.name))
    metrics_eval = {}
    metrics_test = {}    
    if dataset_eval:
        dataset_eval = preprocess_dataset(nli_dataset_info, dataset_eval, tokenizer)
        metrics_eval = trainer.evaluate(eval_dataset=dataset_eval)
    else:
        print('Skipped eval.')
    
    if dataset_test:
        dataset_test = preprocess_dataset(nli_dataset_info, dataset_test, tokenizer)
        metrics_test = trainer.evaluate(eval_dataset=dataset_test)
    else:
        print('Skipped test.')

    results = \
        { 
            nli_dataset_info.name: 
                { 
                    'eval': metrics_eval, 
                    'test': metrics_test 
                }
        }

    print('Results on {}:'.format(nli_dataset_info.name))
    print(json.dumps(results, indent=4))
    
    return results


#######################################################################################################


def split_dataset(
      nli_dataset_info: NLIDatasetInfo
    , dataset: DatasetDict
) \
    -> Tuple[Dataset, Dataset, Dataset]:

    dataset_train = concatenate_datasets([dataset[split] for split in nli_dataset_info.train_splits]) \
                    if nli_dataset_info.train_splits != [] else None
    dataset_eval = concatenate_datasets([dataset[split] for split in nli_dataset_info.eval_splits]) \
                    if nli_dataset_info.eval_splits != [] else None
    dataset_test = concatenate_datasets([dataset[split] for split in nli_dataset_info.test_splits]) \
                    if nli_dataset_info.test_splits != [] else None
    return dataset_train, dataset_eval, dataset_test


def preprocess_dataset(
      nli_dataset_info: NLIDatasetInfo
    , dataset: DatasetDict
    , tokenizer: PreTrainedTokenizer
) \
    -> DatasetDict:

    tokenize_batch = lambda batch: tokenize_pre_and_hypo(nli_dataset_info, batch, tokenizer)
    filter_batch = lambda batch: filter_labels_available(nli_dataset_info, batch)

    return dataset.map(tokenize_batch, batched=True, num_proc=4) \
                  .filter(filter_batch, batched=True, num_proc=4)


def tokenize_pre_and_hypo(
      nli_dataset_info: NLIDatasetInfo
    , batch: Dict[str, List]
    , tokenizer: PreTrainedTokenizer 
):
    return tokenizer(
              text=batch[nli_dataset_info.pre_col]
            , text_pair=batch[nli_dataset_info.hypo_col]
            , truncation=True
            , max_length=tokenizer.model_max_length
            , padding=False
            , return_attention_mask=True
            , return_token_type_ids=True
        )


def filter_labels_available(
      nli_dataset_info: NLIDatasetInfo
    , batch: Dict[str, list]
):
    return numpy.isin(batch[nli_dataset_info.label_col], nli_dataset_info.labels)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else \
            p.predictions
    preds = numpy.argmax(preds, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
              y_true=p.label_ids
            , y_pred=preds
            , average='macro'
            , zero_division=0.0
        )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


#######################################################################################################


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
              '--do_train'
            , action='store_true'
            , help='set this flag to train the model. '
                   'if not set, the training step will be skipped'
        )
    args = parser.parse_args()
    main(args._get_args)
