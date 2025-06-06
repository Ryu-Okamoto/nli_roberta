# train and test NLI model whose base-model is RoBERTa-large
# ref: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py

import os
from dataclasses import dataclass
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


#######################################################################################################


def main():

    ## model setup
    tokenizer, model = load_pretrained_model(
              pretrained_model_name='roberta-large'
            , classlabel_list=['entailment', 'neutral', 'contradiction']
        )

    ## configure training
    use_mixed_precision = True and torch.cuda.is_available()
    training_args = TrainingArguments(
              output_dir='tmp_output/'          # reset after
            , overwrite_output_dir=True         # to overwrite the output directory
            , do_train=True
            , do_eval=True
            , eval_strategy='epoch'             # to evaluate every epoch
            , save_strategy='epoch'             # to save the model every epoch
            , logging_strategy='epoch'          # to log every epoch
            , learning_rate=1e-5                # equivalent to DocNLI
            , weight_decay=1e-2                 # to regularize
            , num_train_epochs=5                # equivalent to DocNLI
            , per_device_train_batch_size=16
            , gradient_accumulation_steps=2     # batch_size ~ this * per_device_train_epoch_batch_size
            , per_device_eval_batch_size=16
            , fp16=use_mixed_precision          # to use mixed precision training
            , load_best_model_at_end=True       # to select best model checkpoint(epoch) for next trainee
            , metric_for_best_model='accuracy'  # metric to determine best model checkpoint
        )

    ## train on each dataset
    tokenizer_1, model_1 = train(
              nli_dataset_info=SNLI_INFO
            , tokenizer=tokenizer
            , model=model
            , training_args=training_args
        )
    tokenizer_2, model_2 = train(
              nli_dataset_info=MNLI_INFO
            , tokenizer=tokenizer_1
            , model=model_1
            , training_args=training_args
        )
    tokenizer_3, model_3 = train(
              nli_dataset_info=ANLI_INFO
            , tokenizer=tokenizer_2
            , model=model_2
            , training_args=training_args
        )
    
    ## test on each dataset
    test_results = []
    test_results.append(test(SNLI_INFO, tokenizer_3, model_3))
    test_results.append(test(MNLI_INFO, tokenizer_3, model_3))
    test_results.append(test(ANLI_INFO, tokenizer_3, model_3))
    print(test_results)


#######################################################################################################


def load_pretrained_model(
      pretrained_model_name: str
    , classlabel_list: list[Any]
) \
    -> tuple[PreTrainedTokenizer, PreTrainedModel]:

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
    training_args.output_dir = os.path.join(OUTPUT_DIR, nli_dataset_info.name)
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
    -> Tuple[Dict[str, Dict[str, float]]]:

    if nli_dataset_info.test_splits == []:
        print('Skip testing on {}.'.format(nli_dataset_info.name))
        return {}

    print('Prepare test set for {}...'.format(nli_dataset_info.name))
    dataset = load_dataset(nli_dataset_info.hf_path, cache_dir=DATASET_CACHE_DIR)
    _, _, dataset_test = split_dataset(nli_dataset_info, dataset, only_test=True)

    dataset_test = preprocess_dataset(nli_dataset_info, dataset_test, tokenizer)

    trainer = Trainer(
              model=model
            , compute_metrics=compute_metrics
            , processing_class=tokenizer
        )

    print(f'Running test on {nli_dataset_info.name}...')
    metrics = trainer.evaluate(eval_dataset=dataset_test)
    print(f"Test results on {nli_dataset_info.name}: {metrics}")
    
    return { nli_dataset_info.name: metrics }


#######################################################################################################


def split_dataset(
      nli_dataset_info: NLIDatasetInfo
    , dataset: DatasetDict
    , only_test: bool = False
) \
    -> Tuple[Dataset, Dataset, Dataset]:

    dataset_train = concatenate_datasets([dataset[split] for split in nli_dataset_info.train_splits]) \
                    if nli_dataset_info.train_splits != [] and not only_test else None
    dataset_eval = concatenate_datasets([dataset[split] for split in nli_dataset_info.eval_splits]) \
                    if nli_dataset_info.eval_splits != [] and not only_test else None
    dataset_test = concatenate_datasets([dataset[split] for split in nli_dataset_info.test_splits]) \
                    if nli_dataset_info.test_splits != [] and only_test else None
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
    acc = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
              y_true=p.label_ids
            , y_pred=preds
            , average='macro'
            , zero_division=0.0
        )
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


#######################################################################################################


if __name__ == '__main__':
    main()
