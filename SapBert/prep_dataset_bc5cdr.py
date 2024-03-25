import os

import datasets

from datasets import load_dataset

from datasets import ClassLabel


datasets.logging.set_verbosity(10)
import logging

logging.basicConfig(filename='./example.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


""" NER dataset compiled by T-NER library https://github.com/asahi417/tner/tree/master/tner """
import json
from itertools import chain
import datasets

logger = datasets.logging.get_logger(__name__)
_DESCRIPTION = """[Bio Creative 5 CDR NER dataset](https://academic.oup.com/database/article/doi/10.1093/database/baw032/2630271?login=true)"""
_NAME = "bc5cdr"
_VERSION = "1.0.0"
_CITATION = """
@article{wei2016assessing,
  title={Assessing the state of the art in biomedical relation extraction: overview of the BioCreative V chemical-disease relation (CDR) task},
  author={Wei, Chih-Hsuan and Peng, Yifan and Leaman, Robert and Davis, Allan Peter and Mattingly, Carolyn J and Li, Jiao and Wiegers, Thomas C and Lu, Zhiyong},
  journal={Database},
  volume={2016},
  year={2016},
  publisher={Oxford Academic}
}
"""

_HOME_PAGE = "https://github.com/asahi417/tner"
_URL = f'https://huggingface.co/datasets/tner/{_NAME}/raw/main/dataset'
_URLS = {
    str(datasets.Split.TEST): [f'{_URL}/test.json'],
    str(datasets.Split.TRAIN): [f'{_URL}/train.json'],
    str(datasets.Split.VALIDATION): [f'{_URL}/valid.json'],
}


class BC5CDRConfig(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """BuilderConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BC5CDRConfig, self).__init__(**kwargs)


class BC5CDR(datasets.GeneratorBasedBuilder):
    """Dataset."""

    BUILDER_CONFIGS = [
        BC5CDRConfig(name=_NAME, version=datasets.Version(_VERSION), description=_DESCRIPTION),
    ]

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download_and_extract(_URLS)
        return [datasets.SplitGenerator(name=i, gen_kwargs={"filepaths": downloaded_file[str(i)]})
                for i in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]]

    def _generate_examples(self, filepaths):
        _key = 0
        for filepath in filepaths:
            logger.info(f"generating examples from = {filepath}")
            with open(filepath, encoding="utf-8") as f:
                _list = [i for i in f.read().split('\n') if len(i) > 0]
                labels_map = {0: "O", 1: "B-Chemical", 2: "B-Disease", 3: "I-Disease", 4: "I-Chemical"}
                for i in _list:
                    data = json.loads(i)
                    yield _key, {
                        "tokens":data.get("tokens"),
                        "tags": [labels_map[x] for x in data.get("tags")]
                    }
                    _key += 1

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-Chemical",
                                "B-Disease",
                                "I-Disease",
                                "I-Chemical"
                            ]
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage=_HOME_PAGE,
            citation=_CITATION,
        )

class BC5CDRDataset(object):
    NAME = "BC5CDRDataset"

    def __init__(self):
        self._dataset = BC5CDR()
        self._dataset.download_and_prepare()
        self._dataset = self._dataset.as_dataset()

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['tags'].feature.names

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
    hf_dataset = BC5CDRDataset()
    dataset = hf_dataset.dataset

    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])
    print(hf_dataset.labels)

    print("List of tags: ", dataset['train'].features['tags'].feature.names)


    print("First sample: ", dataset['train'][0])