import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib

# Définir le nom de l'expérience MLflow
mlflow.set_experiment("Analyse de Sentiments Twitter")

def train_model(model_name, pipeline):
    """Entraîne un modèle et log les informations avec MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Charger les données
        train_df = pd.read_csv(os.path.join('data', 'train.csv'))
        X_train = train_df['text'].astype(str)
        y_train = train_df['sentiment']

        # Logger les paramètres du modèle
        params = pipeline.get_params()
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tfidf_ngram_range", params['tfidf__ngram_range'])
        mlflow.log_param("tfidf_max_features", params['tfidf__max_features'])
        if model_name == 'LogisticRegression':
            mlflow.log_param("clf_max_iter", params['clf__max_iter'])

        # Entraînement
        print(f"Entraînement du modèle {model_name}...")
        pipeline.fit(X_train, y_train)
        print("Entraînement terminé.")

        # Logger le modèle dans MLflow
        mlflow.sklearn.log_model(pipeline, f"{model_name}_pipeline")

        # ➕ Sauvegarde locale du modèle
        os.makedirs('models', exist_ok=True)
        local_path = os.path.join('models', f"{model_name.lower().replace(' ', '_')}_pipeline.joblib")
        joblib.dump(pipeline, local_path)
        print(f"Modèle {model_name} sauvegardé localement -> {local_path}")

if __name__ == "__main__":
    # Pipeline pour la Régression Logistique
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    train_model('LogisticRegression', lr_pipeline)

    # Pipeline pour Naive Bayes
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])
    train_model('NaiveBayes', nb_pipeline)
