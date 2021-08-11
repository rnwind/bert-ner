# -*- coding: utf-8 -*-
from src.bert_model import NerModel

if __name__ == '__main__':
    dataset_path = 'resources/JSON for 2nd task.json'
    model = NerModel(dataset_path)
    model.train_model()

