# -*- coding: utf-8 -*-
from src.ner_model import NerModel

if __name__ == '__main__':
    dataset_path = 'resources/JSON for 2nd task.json'
    model_path = 'model/spacy_ner_model'
    model = NerModel(dataset_path, model_path)

    # Uncomment to train the model on train data (~ 5 min on CPU)
    # model.train_model()

    # Load trained model from disk and evaluates it
    model.load_and_eval()
