# -*- coding: utf-8 -*-
from src.bert_model import BertDataset

if __name__ == '__main__':
    dataset_path = 'resources/JSON for 2nd task.json'
    dataset = BertDataset(dataset_path)
