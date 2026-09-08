from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import copy
from catboost import Pool
import traceback

app = FastAPI()

# --- 1. CONFIGURATION CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CHARGEMENT DES MODÈLES ---
print("Chargement du modèle CatBoost et de l'imputer...")
modele = joblib.load('best_catboost_model.pkl')
imputer = joblib.load('knn_imputer.pkl')

# Récupération automatique des variables catégoriques du modèle
try:
    cat_indices = modele.get_cat_feature_indices()
    cat_features_names = [modele.feature_names_[i] for i in cat_indices]
    print(f"Features catégoriques détectées : {cat_features_names}")
except Exception as e:
    print(f"Avertissement (features catégoriques non trouvées) : {e}")
    # FALLBACK : Si le modèle ne stocke pas les noms, listez-les manuellement ici
    # cat_features_names = ["sexe", "hypertension", "diabete"]
    cat_features_names = [] 

# --- 3. CHARGEMENT ET PIVOT DU CSV ---
try:
    df_long = pd.read_csv("DataTest_long.csv")
    df_patients = df_long.pivot_table(
        index="subject_id",
        columns="variable",
        values="Value",
        aggfunc="first"
    )
    df_patients = df_patients.apply(pd.to_numeric, errors='coerce')
    print("Données patients pivotées et chargées avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement ou du pivot du CSV : {e}")

# --- FONCTION HELPER CENTRALE ---
def prepare_for_catboost(df_raw):
    """
    1. Sépare les données pour le KNN Imputer.
    2. Impute les données manquantes.
    3. Reconstruit le DataFrame COMPLET avec toutes les features attendues par CatBoost.
    4. Gère les types pour l'inférence.
    """
    # --- 1. SOUS-ENSEMBLE POUR LE KNN IMPUTER ---
    if hasattr(imputer, "feature_names_in_"):
        df_for_imputer = df_raw.reindex(columns=imputer.feature_names_in_)
    else:
        df_for_imputer = df_raw.copy()

    # Le KNNImputer exige des floats purs, on force la conversion
    df_for_imputer = df_for_imputer.apply(pd.to_numeric, errors='coerce').astype(float)

    # Imputation
    array_imputed = imputer.transform(df_for_imputer)
    df_imputed = pd.DataFrame(array_imputed, columns=df_for_imputer.columns)

    # --- 2. RECONSTRUCTION DU DATAFRAME COMPLET POUR CATBOOST ---
    # On récupère toutes les colonnes requises par le modèle
    model_features = modele.feature_names_
    
    for col in model_features:
        if col not in df_imputed.columns:
            # Si la colonne n'est pas passée par le KNN (ex: nihss_score), 
            # on la récupère de la requête front-end d'origine
            if col in df_raw.columns:
                df_imputed[col] = df_raw[col].values
            else:
                # Si elle est totalement absente du payload, on met NaN pour CatBoost
                df_imputed[col] = np.nan 

    # Alignement final strict sur l'ordre exact attendu par CatBoost
    df_final = df_imputed.reindex(columns=model_features)
    
    # --- 3. RESTAURATION DES TYPES CATÉGORIQUES ---
    for col in cat_features_names:
        if col in df_final.columns:
            # Conversion robuste en chaîne de caractères pour les features catégoriques
            df_final[col] = df_final[col].fillna(-1).round().astype(int).astype(str)
            df_final[col] = df_final[col].replace('-1', 'NaN')
            
    return df_final


# --- 4. ROUTES DE L'API ---

@app.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    """Renvoie les données cliniques d'un patient existant"""
    if patient_id not in df_patients.index:
        raise HTTPException(status_code=404, detail="Patient introuvable")
    
    patient_ligne = df_patients.loc[patient_id]
    if isinstance(patient_ligne, pd.DataFrame):
        patient_ligne = patient_ligne.iloc[0]
        
    patient_ligne = patient_ligne.replace({np.nan: None})
    patient_data = patient_ligne.to_dict()
    return {"donnees": patient_data}

@app.post("/predict")
def predict_mrs(patient: dict):
    """Prédit le mRS avec une préparation robuste pour CatBoost"""
    try:
        df_nouveau = pd.DataFrame([patient])
        
        # Utilisation de la fonction de préparation centralisée
        df_imputed = prepare_for_catboost(df_nouveau)
        
        # Création du Pool CatBoost sécurisé
        eval_pool = Pool(data=df_imputed, cat_features=cat_features_names)
        prediction_brute = modele.predict(eval_pool)
        
        mrs_final = int(np.clip(np.round(prediction_brute[0]), 0, 6))
        
        return {"mRS_predit": mrs_final, "risque_brut": float(prediction_brute[0])}
    
    except Exception as e:
        print("\n" + "="*50)
        print(f"🔥 ERREUR CRITIQUE dans /predict : {e}")
        traceback.print_exc() 
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate_curve/{variable}")
def simulate_curve(variable: str, min_val: float, max_val: float, patient: dict):
    """Génère les 50 points (X, Y) de la courbe de risque CatBoost de manière stable"""
    try:
        valeurs_x = np.linspace(min_val, max_val, 50)
        
        donnees_clones = []
        for val in valeurs_x:
            clone = copy.deepcopy(patient)
            clone[variable] = val
            donnees_clones.append(clone)
            
        df_clones = pd.DataFrame(donnees_clones)
        
        # Préparation vectorisée pour les 50 clones
        df_imputed = prepare_for_catboost(df_clones)
        
        # Création du Pool CatBoost sécurisé
        eval_pool = Pool(data=df_imputed, cat_features=cat_features_names)
        predictions_brutes = modele.predict(eval_pool)
        
        courbe = [{"x": float(valeurs_x[i]), "y": float(predictions_brutes[i])} for i in range(50)]
        return {"courbe": courbe}
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"🔥 ERREUR CRITIQUE dans /simulate_curve : {e}")
        traceback.print_exc() 
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))