import wandb 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def evaluation(model, X, y, LABELS, mode):

    y_pred = model.predict(X)

    # classification performance results
    report = classification_report(y, y_pred, target_names=LABELS, zero_division=np.nan, output_dict=True)

    # log results
    wandb.log({f"report_{mode}": report})
    wandb.log({
        f"{mode}_macro_f1": report['macro avg']['f1-score'],
        f"{mode}_weighted_f1": report['weighted avg']['f1-score']
    })


def train(X, y, LABELS):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # inference on the trained model
    evaluation(nb_model, X_train, y_train, LABELS, "train")
    evaluation(nb_model, X_test, y_test, LABELS, "test")
