# -*- coding: utf-8 -*-
from src.ner_model import NerModel

if __name__ == '__main__':
    dataset_path = 'resources/JSON for 2nd task.json'
    model = NerModel(dataset_path)

    # Train the model on train data (~ 5 min on CPU)
    model.train_model()

    # Load trained model from disk and evaluates it
    model.load_and_eval()

