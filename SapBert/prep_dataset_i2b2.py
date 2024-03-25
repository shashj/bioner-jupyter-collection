import os

import datasets

from datasets import ClassLabel


datasets.logging.set_verbosity(10)
import logging

logging.basicConfig(filename='./example.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

_CITATION = """\
None right now
"""

_DESCRIPTION = """\
i2b2 2006
"""

_TRAINING_FILE = "../datasets/i2b2/PHI_Processed_data/train.txt"
_DEV_FILE = "../datasets/i2b2/PHI_Processed_data/dev.txt"
_TEST_FILE = "../datasets/i2b2/PHI_Processed_data/test.txt"

class i2b2deid2006Config(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forConll2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(i2b2deid2006Config, self).__init__(**kwargs)


class i2b2deid2006(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        i2b2deid2006Config(name="i2b2deid2006", version=datasets.Version("1.0.0"), description="i2b2 deid 2006 dataset"),
    ]

    def __init__(self,
                 *args,
                 cache_dir='./',
                 train_file="train.txt",
                 val_file="dev.txt",
                 test_file="test.txt",
                 ner_tags=("O",   "I-PHONE",   "I-PATIENT",   "I-LOCATION",   "I-ID",   "I-HOSPITAL",   "I-DOCTOR",   "I-DATE",   "B-PHONE",   "B-PATIENT",   "B-LOCATION",   "B-ID",   "B-HOSPITAL",   "B-DOCTOR",   "B-DATE", "B-AGE"),
                 **kwargs):
        self._ner_tags = ner_tags
        self._train_file = train_file
        self._val_file = val_file
        self._test_file = test_file
        super(i2b2deid2006, self).__init__(*args, cache_dir=cache_dir, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._ner_tags))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
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
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]
    
    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # i2b2 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }

class i2b2deid2006Dataset(object):
    NAME = "i2b2deid2006Dataset"

    def __init__(self):
        self._dataset = i2b2deid2006()
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
    dataset = i2b2deid2006Dataset().dataset

    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])

    print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)


    print("First sample: ", dataset['train'][11])