import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report
import mlflow

if __name__ == "__main__":
    # Même expérience que l'entraînement
    mlflow.set_experiment("Analyse de Sentiments Twitter")

    # Charger les données de test
    test_df = pd.read_csv(os.path.join('data', 'test.csv'))
    X_test = test_df['text'].astype(str)
    y_test = test_df['sentiment']

    # --- Éval Logistic Regression ---
    print("Évaluation du modèle de Régression Logistique...")
    lr_pipeline = joblib.load(os.path.join('models', 'logistic_regression_pipeline.joblib'))
    lr_predictions = lr_pipeline.predict(X_test)

    lr_report = classification_report(
        y_test, lr_predictions, target_names=['Négatif', 'Positif'], output_dict=True
    )
    print("\n--- Rapport de Classification (Régression Logistique) ---")
    print(classification_report(y_test, lr_predictions, target_names=['Négatif', 'Positif']))

    with mlflow.start_run(run_name="LogisticRegression-eval"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", lr_report["accuracy"])
        mlflow.log_metric("f1_weighted", lr_report["weighted avg"]["f1-score"])
        # Sauvegarde du rapport texte en artefact
        os.makedirs("artifacts", exist_ok=True)
        lr_txt = os.path.join("artifacts", "lr_classification_report.txt")
        with open(lr_txt, "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, lr_predictions, target_names=['Négatif', 'Positif']))
        mlflow.log_artifact(lr_txt, artifact_path="reports")

    # --- Éval Naive Bayes ---
    print("\nÉvaluation du modèle Naive Bayes...")
    nb_pipeline = joblib.load(os.path.join('models', 'naive_bayes_pipeline.joblib'))
    nb_predictions = nb_pipeline.predict(X_test)

    nb_report = classification_report(
        y_test, nb_predictions, target_names=['Négatif', 'Positif'], output_dict=True
    )
    print("\n--- Rapport de Classification (Naive Bayes) ---")
    print(classification_report(y_test, nb_predictions, target_names=['Négatif', 'Positif']))

    with mlflow.start_run(run_name="NaiveBayes-eval"):
        mlflow.log_param("model_type", "NaiveBayes")
        mlflow.log_metric("accuracy", nb_report["accuracy"])
        mlflow.log_metric("f1_weighted", nb_report["weighted avg"]["f1-score"])
        # Sauvegarde du rapport texte en artefact
        nb_txt = os.path.join("artifacts", "nb_classification_report.txt")
        with open(nb_txt, "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, nb_predictions, target_names=['Négatif', 'Positif']))
        mlflow.log_artifact(nb_txt, artifact_path="reports")

    # --- Tableau comparatif en console (facultatif) ---
    results = {
        "Modèle": ["Régression Logistique", "Naive Bayes"],
        "Accuracy": [lr_report['accuracy'], nb_report['accuracy']],
        "F1-Score (Pondéré)": [
            lr_report['weighted avg']['f1-score'],
            nb_report['weighted avg']['f1-score']
        ]
    }
    results_df = pd.DataFrame(results)
    print("\n--- Tableau Comparatif des Performances ---")
    print(results_df.to_string(index=False))
