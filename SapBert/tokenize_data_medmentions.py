from datasets import load_metric
import datasets
from transformers import AutoTokenizer
from datasets import load_dataset

from prep_dataset_medmentions import MedMentionsDataset

import logging

logging.basicConfig(filename='./example.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class HFTokenizer(object):
    NAME = "HFTokenizer"

    def __init__(self,
                 hf_pretrained_tokenizer_checkpoint):
        self._tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_tokenizer_checkpoint)

    @property
    def tokenizer(self):
        return self._tokenizer

    @staticmethod
    def init_vf(hf_pretrained_tokenizer_checkpoint):
        return HFTokenizer(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    def tokenize_and_align_labels(self,
                                  examples,
                                  label_all_tokens=True):
        tokenized_inputs = self._tokenizer(examples["tokens"],
                                           truncation=True,
                                           is_split_into_words=True, max_length=512)
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

if __name__ == '__main__':

    hf_pretrained_tokenizer_checkpoint = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    dataset = MedMentionsDataset().dataset

    hf_preprocessor = HFTokenizer.init_vf(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    tokenized_datasets = dataset.map(hf_preprocessor.tokenize_and_align_labels, batched=True)

    print(dataset)

    print("*" * 100)

    print(tokenized_datasets)

    print("First sample: ", dataset['train'][0])

    print("*" * 100)

    print("First tokenized sample: ", tokenized_datasets['train'][0])