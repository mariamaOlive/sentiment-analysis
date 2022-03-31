import optuna
from sklearn.metrics import f1_score
from functools import partial
from sklearn.ensemble import RandomForestClassifier

def get_best_parameters(X_train_vec, X_valid_vec, y_train, y_valid):
    study = optuna.create_study(direction="maximize")
    obj_parcial = partial(objective, X_train_vec, X_valid_vec, y_train, y_valid)
    study.optimize(obj_parcial, n_trials=20)

    return study.best_params


def objective(X_train_vec, X_valid_vec, y_train, y_valid, trial):

    param_grid = {"n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                  "max_depth": trial.suggest_int("max_depth", 2, 128, log=True),
                  "criterion": trial.suggest_categorical('criterion', ["gini", "entropy"])}


    forest = RandomForestClassifier(**param_grid) 
    forest = forest.fit(X_train_vec, y_train)
    preds = forest.predict(X_valid_vec)

    score = f1_score(y_valid, preds, average=None)[0]
    return score

