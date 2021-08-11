# -*- coding: utf-8 -*-
from pathlib import Path
import random
import json
import re

import numpy as np
import spacy
from spacy.training import Example
from spacy.tokens import Doc


class NerModel:
    def __init__(self, dataset_path):
        self.max_len = 256
        self.random_seed = 42
        self.test_ratio = 0.1
        self.dataset_size = None
        self.max_item_len = 0
        self.tag2id = None
        self.id2tag = None
        self.unique_tags = None
        self._model = None
        self._dataset_path = Path(dataset_path)
        self._model_path = Path('model/spacy_ner_model')
        self._data = self._load_data(self._dataset_path)
        self.train_data, self.test_data = self._process_data()
        # self.train_model(train_dataset, test_dataset)
        # self.test_model(test_dataset)

    def _load_data(self, data_path):
        texts = list()
        labels = list()
        with data_path.open('r') as f:
            data = json.load(f)
        self.dataset_size = len(data)
        print(f'Data loaded: {self.dataset_size} items')
        return data

    def _process_data(self):
        data = self._clean_extra_spaces_in_labels()
        # spacy_data = self._trim_entity_spans(data)
        train_data, test_data = self.train_test_split(data)
        return train_data, test_data






    def train_test_split(self, data):
        all_items = data.copy()
        random.Random(self.random_seed).shuffle(all_items)
        train_size = self.dataset_size - int(self.dataset_size * self.test_ratio) - 1
        return all_items[:train_size], all_items[train_size:]

    def create_eval_examples(self, model):
        test_examples = list()
        for text, annotation in self.test_data:
            doc = model.make_doc(text)
            example = Example.from_dict(doc, annotation)
            test_examples.append(example)
        return test_examples

    def train_model(self):
        nlp = spacy.blank('de')
        ner = None
        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner', last=True)

        for _, annotation in self.train_data:
            for ent in annotation['entities']:
                ner.add_label(ent[2])

        test_examples = self.create_eval_examples(nlp)
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            best_score = 0
            for i in range(15):
                print(f'For epoch: {i}\n')
                random.shuffle(self.train_data)
                losses = {}
                index = 0

                # train
                for text, annotation in self.train_data:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotation)
                    nlp.update(
                        [example],
                        drop=0.2,
                        sgd=optimizer,
                        losses=losses)
                print('loss:', losses['ner'])

                # eval and save
                scores = nlp.evaluate(examples=test_examples, batch_size=8)
                print('Scores on test:', scores)
                if scores['ents_f'] > best_score:
                    best_score = scores['ents_f']
                    self._save_model(nlp)

    def _save_model(self, model):
        self._model_path.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._model.to_disk(self._model_path)
        print('model saved to disk')

    def load_and_eval(self):
        print('Loading model from disk...')
        self._model = spacy.load(self._model_path)
        print('Ok')

        test_examples = self.create_eval_examples(self._model)
        scores = self._model.evaluate(examples=test_examples, batch_size=8)
        print(f'\nEval scores: {scores}')

        print(f'\n\n{"="*80} All predictions on the test dataset {"="*80}\n')
        for text, annotation in self.test_data:
            pred = self._model(text)
            print('text:', text)
            print('Predictions: ')
            for ent in pred.ents:
                print(f"{ent.label_.upper()} - {ent.text}")
            print('\nTrue entities: ')
            for ent in annotation['entities']:
                print(f"{ent[2].upper()} - {text[ent[0]:ent[1]]}")
            print('-'*100)

    def _clean_extra_spaces_in_labels(self):
        training_data = list()
        try:
            for item in self._data:
                text = item['text'].replace("\n", " ")
                annotations = item['labels']
                entities = list()
                for annotation in annotations:
                    point_start = annotation[0]
                    point_end = annotation[1]
                    point_text = text[point_start:point_end]
                    point_label = annotation[2]
                    lstrip_diff = len(point_text) - len(point_text.lstrip())
                    rstrip_diff = len(point_text) - len(point_text.rstrip())
                    if lstrip_diff != 0:
                        point_start = point_start + lstrip_diff
                    if rstrip_diff != 0:
                        point_end = point_end - rstrip_diff
                    entities.append((point_start, point_end, point_label))
                training_data.append((text, {"entities": entities}))
            return training_data
        except Exception as e:
            print('Can\'t process dataset: ', e)
            return None

    def _trim_entity_spans(self, data: list) -> list:
        """Removes leading and trailing white spaces from entity spans.

        Args:
            data (list): The data to be cleaned in spaCy JSON format.

        Returns:
            list: The cleaned data.
        """
        invalid_span_tokens = re.compile(r'\s')

        cleaned_data = list()
        for text, annotations in data:
            entities = annotations['entities']
            valid_entities = []
            for start, end, label in entities:
                valid_start = start
                valid_end = end
                while valid_start < len(text) and invalid_span_tokens.match(
                        text[valid_start]):
                    # print(invalid_span_tokens.match(text[valid_start]))
                    valid_start += 1
                while valid_end > 1 and invalid_span_tokens.match(
                        text[valid_end - 1]):
                    # print(invalid_span_tokens.match(text[valid_end - 1]))
                    valid_end -= 1
                valid_entities.append([valid_start, valid_end, label])
            cleaned_data.append([text, {'entities': valid_entities}])
        return cleaned_data
