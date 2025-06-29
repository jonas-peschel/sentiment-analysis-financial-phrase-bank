from task_2_data_preprocessing import data_preprocessing
from task_3_1_train_bayes import train
import wandb 

import gc

def main():

    with wandb.init() as run:

        config = wandb.config

        run.name += "_" + "_".join([f"{key}_{config[key]}" for key in config.keys()])

        data_filepath = "FinancialPhraseBank-v1.0/Sentences_50Agree.txt"
        X, y, LABELS = data_preprocessing(config, data_filepath)

        train(X, y, LABELS)

        # garbage removal
        del X, y, LABELS 
        gc.collect()

if __name__ == "__main__":
    main()

