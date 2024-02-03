import os

import datasets

from datasets import load_dataset

from datasets import ClassLabel

import pandas as pd


datasets.logging.set_verbosity(10)
import logging

logging.basicConfig(filename='./example.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


_CITATION = """\
https://github.com/chanzuckerberg/MedMentions
"""

_DESCRIPTION = """\
Corpus: The MedMentions corpus consists of 4,392 papers (Titles and Abstracts) randomly selected from among papers released on PubMed in 2016, 
that were in the biomedical field, published in the English language, and had both a Title and an Abstract.
"""

_HOMEPAGE = "https://github.com/chanzuckerberg/MedMentions"
_URL = "https://huggingface.co/datasets/ibm/MedMentions-ZS/raw/main/data"
_TRAINING_FILE = "../datasets/medmentions/train-00000-of-00001.parquet"
_DEV_FILE = "../datasets/medmentions/validation-00000-of-00001.parquet"
_TEST_FILE = "../datasets/medmentions/test-00000-of-00001.parquet"


class MedMentionsConfig(datasets.BuilderConfig):
    """BuilderConfig for MedMentions"""

    def __init__(self, **kwargs):
        """BuilderConfig for MedMentions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedMentionsConfig, self).__init__(**kwargs)


class MedMentions(datasets.GeneratorBasedBuilder):
    """MedMentions dataset."""

    BUILDER_CONFIGS = [
        MedMentionsConfig(name="MedMentions", version=datasets.Version("1.0.0"), description="MedMentions dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'B-T058', 'I-T170', 'I-T091', 'I-T058', 'I-T031', 'B-T097', 
                                'I-T098', 'I-T201', 'I-T005', 'B-T092', 'I-T204', 'B-T170', 
                                'I-T097', 'B-T091', 'B-T005', 'B-T033', 'I-T168', 'B-T082', 
                                'B-T062', 'I-T017', 'B-T031', 'I-T007', 'B-T201', 'B-T007', 
                                'B-T022', 'B-T074', 'I-T074', 'O', 'I-T033', 'I-T038', 'I-T022', 
                                'I-T062', 'B-T103', 'I-T103', 'B-T037', 'I-T092', 'B-T204', 'B-T038',
                                'I-T037', 'B-T168', 'B-T017', 'I-T082', 'B-T098'
                            ]
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_files = {
            "train": _TRAINING_FILE,
            "dev": _DEV_FILE,
            "test": _TEST_FILE
        }

        urls_to_download = {
           "train": _TRAINING_FILE,
           "dev": _DEV_FILE,
           "test": _TEST_FILE
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        df = pd.read_parquet(filepath)
        guid = 0
        for index, row in df.iterrows():
            guid+=1
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": list(row['tokens']),
                "ner_tags": list(row['ner_tags']),
            }

class MedMentionsDataset(object):
    NAME = "MedMentionsDataset"

    def __init__(self):
        self._dataset = MedMentions()
        self._dataset.download_and_prepare()
        self._dataset = self._dataset.as_dataset()

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

    def test(self):
        return self._dataset["test"]

    def validation(self):
        return self._dataset["validation"]


if __name__ == '__main__':
    hf_dataset = MedMentionsDataset()
    dataset = hf_dataset.dataset

    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])
    print(hf_dataset.labels)

    print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)


    print("First sample: ", dataset['train'][11])