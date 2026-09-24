"""
Launch :  uvicorn explain_api:app --reload --port 8000
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from catboost import Pool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).parent
CHEMIN_CSV = BASE / "DataTest_long.csv"
CHEMIN_MODELE = BASE / "best_catboost_model.pkl"

SEUILS = {
    "sys_blood_pressure": dict(bornes=[-np.inf, 111, 180, np.inf], right=False,
                               label="Sys Blood Pressure", unite="mmHg",
                               classes=["≤110", "111–179", "≥180"]),
    "dis_blood_pressure": dict(bornes=[-np.inf, 105, np.inf], right=False,
                               label="Dis Blood Pressure", unite="mmHg",
                               classes=["<105", "≥105"]),
    "glucose":            dict(bornes=[-np.inf, 60, 141, np.inf], right=False,
                               label="Glucose", unite="mg/dL",
                               classes=["<60", "60–140", ">140"]),
    "cholesterol":        dict(bornes=[-np.inf, 70, 101, np.inf], right=False,
                               label="Cholesterol", unite="mg/dL",
                               classes=["<70", "70–100", ">100"]),
    "prestroke_mrs":      dict(bornes=[-np.inf, 2, 3, np.inf], right=False,
                               label="Pre-stroke mRS", unite="",
                               classes=["0–1", "2", "3–5"]),
    "onset_to_door":      dict(bornes=[-np.inf, 4.5, 24, np.inf], right=True,
                               label="Onset to Door", unite="min",
                               classes=["≤4.5", "4.5–24", ">24"]),
    "door_to_imaging":    dict(bornes=[-np.inf, 20, np.inf], right=True,
                               label="Door to Imaging", unite="min",
                               classes=["≤20", ">20"]),
    "door_to_needle":     dict(bornes=[-np.inf, 30, 45, 60, np.inf], right=True,
                               label="Door to Needle", unite="min",
                               classes=["≤30", "30–45", "45–60", ">60"]),
}

ANTICOAG_SORTIE = ["discharge_apixaban", "discharge_dabigatran", "discharge_edoxaban",
                   "discharge_rivaroxaban", "discharge_warfarin", "discharge_heparin"]

# ---------------------------------------------------------------- chargement --
modele = pickle.load(open(CHEMIN_MODELE, "rb"))
NOMS = list(modele.feature_names_)
CAT = [NOMS[i] for i in sorted(modele.get_cat_feature_indices())]

# `keep_default_na=False` garde les cellules vides en chaine vide au lieu de NaN :
# un NaN n'est pas serialisable en JSON et faisait tomber la route en erreur 500.
_long = pd.read_csv(CHEMIN_CSV, low_memory=False, dtype={"Value": str},
                    keep_default_na=False)
PATIENTS = {
    sid: {r["variable"]: r["Value"] for _, r in grp.iterrows()}
    for sid, grp in _long.groupby("subject_id")
}


def _nombre(v):
    try:
        f = float(str(v).strip())
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _classe(col, valeur):
    """Valeur brute -> numero de classe, avec le decoupage du modele."""
    cfg = SEUILS[col]
    if valeur is None:
        return None
    etiquettes = list(range(len(cfg["bornes"]) - 1))
    c = pd.cut([valeur], bins=cfg["bornes"], labels=etiquettes, right=cfg["right"])[0]
    return None if pd.isna(c) else int(c)


def _vecteur(brut):
    """Construit la ligne de 45 variables attendue par le modele."""
    ligne = {}
    for nom in NOMS:
        v = _nombre(brut.get(nom))
        if nom in SEUILS:                       # variable discretisee
            v = _classe(nom, v)
        elif nom == "anticoagulant_discharge":
            vals = [_nombre(brut.get(c)) for c in ANTICOAG_SORTIE]
            vals = [x for x in vals if x is not None]
            v = max(vals) if vals else None
        elif nom == "anticoagulant_before_onset":
            v = _nombre(brut.get("before_onset_warfarin"))
        # -1 est la valeur "manquant" utilisee a l'entrainement pour les
        # categorielles ; 0.0 pour les continues, apres imputation KNN.
        ligne[nom] = (int(v) if v is not None else -1) if nom in CAT else (v if v is not None else 0.0)
    X = pd.DataFrame([ligne])[NOMS]
    for c in CAT:
        X[c] = X[c].astype(int)
    return X


def _courbe(X, col):
    """Valeur SHAP et prediction pour chaque classe possible de `col`."""
    n_classes = len(SEUILS[col]["bornes"]) - 1
    valeurs = list(range(n_classes))
    lignes = pd.concat([X] * n_classes, ignore_index=True)
    lignes[col] = valeurs
    lignes[col] = lignes[col].astype(int)

    pool = Pool(lignes, cat_features=CAT)
    shap = modele.get_feature_importance(pool, type="ShapValues")
    j = NOMS.index(col)
    predictions = modele.predict(pool)
    return [
        {"classe": int(c), "shap": round(float(shap[i, j]), 4),
         "mrs": round(float(predictions[i]), 3)}
        for i, c in enumerate(valeurs)
    ]


app = FastAPI(title="XAI Stroke — donnees et explications")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/patients")
def liste_patients():
    return {"patients": sorted(PATIENTS)}


@app.get("/patient/{subject_id}")
def patient(subject_id: str):
    if subject_id not in PATIENTS:
        raise HTTPException(404, "patient inconnu")
    return {"donnees": PATIENTS[subject_id]}


class Simulation(BaseModel):
    """Valeurs brutes modifiees par l'utilisateur, par nom de colonne du registre."""
    valeurs: dict[str, float] = {}


def _explication(subject_id: str, simulation: dict | None = None):
    if subject_id not in PATIENTS:
        raise HTTPException(404, "patient inconnu")
    brut = dict(PATIENTS[subject_id])
    if simulation:
        # On remplace les valeurs BRUTES, la discretisation est refaite ensuite :
        # c'est le meme chemin que pour les donnees reelles, donc pas de
        # divergence possible entre ce qui est simule et ce qui est servi.
        for col, valeur in simulation.items():
            if col in SEUILS:
                brut[col] = valeur
    X = _vecteur(brut)

    pool = Pool(X, cat_features=CAT)
    shap = modele.get_feature_importance(pool, type="ShapValues")
    prediction = float(modele.predict(pool)[0])
    valeur_base = float(shap[0, -1])             # derniere colonne = valeur de base

    variables = {}
    for col, cfg in SEUILS.items():
        brute = _nombre(brut.get(col))
        variables[col] = {
            "label": cfg["label"],
            "unite": cfg["unite"],
            "valeur_brute": brute,
            "classe_actuelle": _classe(col, brute),
            "seuils": [b for b in cfg["bornes"] if np.isfinite(b)],
            "libelles_classes": cfg["classes"],
            "shap_actuel": round(float(shap[0, NOMS.index(col)]), 4),
            "courbe": _courbe(X, col),
        }

    return {
        "patient": subject_id,
        "simule": bool(simulation),
        "prediction": {"mrs": round(prediction, 3), "valeur_base": round(valeur_base, 3)},
        "variables": variables,
    }


@app.get("/explain/{subject_id}")
def explain(subject_id: str):
    """Explication du patient tel qu'il est dans le registre."""
    return _explication(subject_id)


@app.post("/explain/{subject_id}")
def explain_simule(subject_id: str, simulation: Simulation):
    """Meme chose, avec les valeurs modifiees par l'utilisateur."""
    return _explication(subject_id, simulation.valeurs)
