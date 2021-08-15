# -*- coding: utf-8 -*-
from pathlib import Path
import random
import json
import re

import spacy
from spacy.training import Example


class NerModel:
    def __init__(self, dataset_path, model_path):
        self.random_seed = 42
        self.test_ratio = 0.1
        self.dataset_size = None
        self._model = None
        self._dataset_path = Path(dataset_path)
        self._model_path = Path(model_path)
        self._data = self._load_data(self._dataset_path)
        self.train_data, self.test_data = self._process_data()

    def _load_data(self, data_path):
        with data_path.open('r') as f:
            data = json.load(f)
        self.dataset_size = len(data)
        print(f'Data loaded: {self.dataset_size} items')
        return data

    def _process_data(self):
        spacy_data = self._clean_extra_spaces_in_labels()
        spacy_data = self._trim_text_spans(spacy_data)
        train_data, test_data = self._train_test_split(spacy_data)
        return train_data, test_data

    def _train_test_split(self, data):
        all_items = data.copy()
        random.Random(self.random_seed).shuffle(all_items)
        train_size = self.dataset_size - int(self.dataset_size * self.test_ratio) - 1
        return all_items[:train_size], all_items[train_size:]

    def _create_eval_examples(self, model):
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

        test_examples = self._create_eval_examples(nlp)
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            best_score = 0
            for i in range(15):
                print(f'\nEpoch: {i}')
                random.shuffle(self.train_data)
                losses = {}

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
        print(f'\n\n{"="*80} Evaluating the model {self._model_path}{"="*80}\n')
        print('Loading model from disk...')
        self._model = spacy.load(self._model_path)
        print('Ok')

        test_examples = self._create_eval_examples(self._model)
        scores = self._model.evaluate(examples=test_examples, batch_size=8)
        print(f'\nEval scores: {scores}')

        print(f'\n\n{"="*80} All predictions on the test dataset {"="*80}\n')
        for text, annotation in self.test_data:
            pred = self._model(text)
            print('Text:', text)
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
                annotations = sorted(annotations, key=lambda x: x[0])
                entities = list()
                ent_offset = 0
                for annotation in annotations:
                    point_start = annotation[0] + ent_offset
                    point_end = annotation[1] + ent_offset
                    point_text = text[point_start:point_end]
                    point_label = annotation[2]
                    lstrip_diff = len(point_text) - len(point_text.lstrip())
                    rstrip_diff = len(point_text) - len(point_text.rstrip())
                    if lstrip_diff != 0:
                        point_start = point_start + lstrip_diff
                    if rstrip_diff != 0:
                        point_end = point_end - rstrip_diff
                    elif text[min(point_end, len(text)-1)] == '.':
                        text = text[:point_end] + ' ' + text[min(point_end, len(text)-1):]
                        ent_offset += 1
                    entities.append([point_start, point_end, point_label])
                training_data.append([text, {"entities": entities}])
            return training_data
        except Exception as e:
            print('Can\'t process dataset: ', e)
            return None

    def _trim_text_spans(self, data: list) -> list:
        new_data = list()
        for text, annotation in data:
            total_span_len = 0
            spans = list()
            for match in re.finditer(r'\s{2,}', text):
                span = match.span()[1] - match.span()[0]
                spans.append(match.span())
                total_span_len += span - 1
            new_entities = self._shift_entities(spans, annotation['entities'])
            new_text = re.sub(r'\s+', ' ', text)
            new_data.append([new_text, {'entities': new_entities}])
        return new_data

    def _shift_entities(self, spans: list, entities: list) -> list:
        new_entities = [x[:] for x in entities]
        for span in spans:
            for i, ent in enumerate(entities):
                span_len = span[1] - span[0]
                if span[1] <= ent[0]:
                    new_entities[i][0] -= span_len - 1
                if span[1] <= ent[1]:
                    new_entities[i][1] -= span_len - 1
        return new_entities
