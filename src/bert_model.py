# -*- coding: utf-8 -*-
from pathlib import Path
import random
import json
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification
from tqdm import tqdm


class BertDataset:
    def __init__(self, dataset_path):
        self.max_len = 512
        self.random_seed = 42
        self.test_ratio = 0.1
        self.dataset_size = None
        self.max_item_len = 0
        self.tag2id = None
        self.id2tag = None
        self.unique_tags = None
        self._model = None
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
        self._dataset_path = Path(dataset_path)
        self._data = self._load_data(self._dataset_path)
        train_dataset, test_dataset = self._process_data()
        self.train_model(train_dataset, test_dataset)

    def _load_data(self, data_path):
        texts = list()
        labels = list()
        with data_path.open('r') as f:
            data = json.load(f)
        self.dataset_size = len(data)
        print(f'Data loaded: {self.dataset_size} items')
        return data

    def _process_data(self):
        data = self._convert_dataturks_to_spacy()
        data = self._trim_entity_spans(data)
        label_data = self._clean_dataset(data)
        texts = self._get_text(data)
        tags = self._create_tags(label_data)
        tokenized_data = self.tokenize_and_align_labels(texts, tags)

        input_ids = np.array(tokenized_data['input_ids'])
        labels = np.array(tokenized_data['labels'])
        train_items, test_items = self.get_train_test_ids()

        train_dataset = tf.data.Dataset.from_tensor_slices((input_ids[train_items], labels[train_items]))
        train_dataset = train_dataset.shuffle(200).batch(8)

        test_dataset = tf.data.Dataset.from_tensor_slices((input_ids[test_items], labels[test_items]))
        test_dataset = test_dataset.batch(8)

        return train_dataset, test_dataset

    def get_train_test_ids(self):
        all_items = list(range(self.dataset_size))
        random.Random(self.random_seed).shuffle(all_items)
        train_size = self.dataset_size - int(self.dataset_size * self.test_ratio) - 1
        return all_items[:train_size], all_items[train_size:]

    def train_model(self, train_data, test_data):
        self._model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased',
                                                                         num_labels=len(self.unique_tags))
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self._model.compile(optimizer=optimizer, loss=self._model.compute_loss, metrics=['accuracy'])
        self._model.fit(train_data, validation_data=test_data, epochs=3, batch_size=8)

    def _get_text(self, data: list):
        texts = list()
        [texts.append(text) for text, _ in data]
        return texts

    def _convert_dataturks_to_spacy(self):
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

    def _clean_dataset(self, data):
        cleanedDF = pd.DataFrame(columns=["setences_cleaned"])
        sum1 = 0
        for i in tqdm(range(len(data))):
            start = 0
            emptyList = ["Empty"] * len(data[i][0].split())
            numberOfWords = 0
            lenOfString = len(data[i][0])
            strData = data[i][0]
            strDictData = data[i][1]
            lastIndexOfSpace = strData.rfind(' ')
            for i in range(lenOfString):
                if (strData[i] == " " and strData[i + 1] != " "):
                    for k, v in strDictData.items():
                        for j in range(len(v)):
                            entList = v[len(v) - j - 1]
                            if (start >= int(entList[0]) and i <= int(entList[1])):
                                emptyList[numberOfWords] = entList[2]
                                break
                            else:
                                continue
                    start = i + 1
                    numberOfWords += 1
                if (i == lastIndexOfSpace):
                    for j in range(len(v)):
                        entList = v[len(v) - j - 1]
                        if (lastIndexOfSpace >= int(entList[0]) and lenOfString <= int(entList[1])):
                            emptyList[numberOfWords] = entList[2]
                            numberOfWords += 1
            cleanedDF = cleanedDF.append(pd.Series([emptyList], index=cleanedDF.columns), ignore_index=True)
            sum1 = sum1 + numberOfWords
        return cleanedDF

    def _create_tags(self, data:pd.DataFrame):
        self.unique_tags = set(data['setences_cleaned'].explode().unique())
        self.tag2id = {tag: id for id, tag in enumerate(self.unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

        labels = data['setences_cleaned'].values.tolist()
        tags = pad_sequences([[self.tag2id.get(l) for l in lab] for lab in labels],
                             maxlen=self.max_len, value=self.tag2id["Empty"], padding="post",
                             dtype="long", truncating="post")
        return tags

    def tokenize_and_align_labels(self, examples, tags, label_all_tokens=True):
        tokenized_inputs = self.tokenizer(examples, truncation=True, is_split_into_words=False, padding='max_length',
                                          max_length=self.max_len)
        labels = []
        for i, label in enumerate(tags):
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

