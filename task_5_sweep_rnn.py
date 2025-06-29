from task_2_data_preprocessing import data_preprocessing
from task_5_train_rnn import train
import wandb 

def main():

    with wandb.init() as run:

        config = wandb.config

        run.name += "_" + "_".join([f"{key}_{config[key]}" for key in config.keys()])

        data_filepath = "FinancialPhraseBank-v1.0/Sentences_50Agree.txt"
        X, y, LABELS = data_preprocessing(config, data_filepath)

        train(config, X, y, LABELS)

if __name__ == "__main__":
    main()

