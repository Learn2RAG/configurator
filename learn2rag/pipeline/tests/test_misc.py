import unittest

from ..embeddings import create_embeddings


class PipelineMiscTestCase(unittest.TestCase):
    def test_embeddings(self) -> None:
        input_sample = [
            'Paderborn is a city in eastern North Rhine-Westphalia, Germany, capital of the Paderborn district.',
            'Bielefeld is a city in the Ostwestfalen-Lippe Region in the north-east of North Rhine-Westphalia, Germany.',
        ]
        model_name = 'sentence-transformers/all-mpnet-base-v2'
        embeddings = create_embeddings(input_sample, model_name)
        assert len(embeddings) == len(input_sample)

    # TODO: actual tests
