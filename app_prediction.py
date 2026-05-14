import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# ---- Initialisation session state ----
if "submitted" not in st.session_state:
    st.session_state.submitted = False

def on_nb_produits_change():
    pass  # la logique est gérée dans le rendu via la clé session

def submit():
    st.session_state.submitted = True

# ---- Configuration ----
st.set_page_config(
    page_title="Prédiction Churn - ATB",
    layout="centered"
)

# ---- Chemin des fichiers ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Donnes")
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_churn_calibrated_best.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_churn.pkl")
FEAT_COLS_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

# ---- Chargement des données et artefacts ----
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "churn_merged.csv"), sep=";")
    return df

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feat_cols = joblib.load(FEAT_COLS_PATH)
    scaler_cols = list(scaler.feature_names_in_)
    return model, scaler, feat_cols, scaler_cols

@st.cache_resource
def load_scaler_cols():
    scaler = joblib.load(SCALER_PATH)
    return list(scaler.feature_names_in_)

@st.cache_data
def build_encoding_maps(df):
    """Construit les mappings Target Encoding depuis les données"""
    target_cols = ['GROUPE_PRODUIT', 'TYPE_COMPTE', 'CATEGORIE_COMPTE']
    maps = {}
    for col in target_cols:
        maps[col] = df.groupby(col)['CHURN'].mean().to_dict()
        # Ajouter la moyenne globale comme fallback
        maps[col + '_global_mean'] = df['CHURN'].mean()
    return maps

@st.cache_data
def get_label_encoder():
    """Mapping DOSSIER_COMPLET"""
    return {"YES": 1, "NO": 0, "OUI": 1, "NON": 0}

@st.cache_data
def get_default_categorie():
    """Mapping TYPE_COMPTE -> CATEGORIE_COMPTE la plus frequente"""
    tc = df.groupby('TYPE_COMPTE')['CATEGORIE_COMPTE'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.median()
    ).to_dict()
    return tc

# ---- Chargement ----
with st.spinner("Chargement du modèle..."):
    df = load_data()
    model, scaler, feat_cols, scaler_cols = load_model()
    encoding_maps = build_encoding_maps(df)
    le_map = get_label_encoder()
    default_categorie = get_default_categorie()



# ---- Titre ----
st.title("Prédiction du Churn Client — ATB")
st.markdown("Testez le modèle XGBoost pour estimer la probabilité de churn d'un client")

st.markdown("---")

# ---- Formulaire de saisie ----
st.subheader("Caractéristiques du client")

col1, col2 = st.columns(2)

with col1:
    nb = st.session_state.get("nb_produits", 1)

    groupe_produit = st.selectbox(
        "Groupe produit",
        options=sorted(df['GROUPE_PRODUIT'].dropna().unique()),
        disabled=(nb == 0),
        help="Groupe de produits auquel appartient le compte" if nb > 0 else "Aucun produit — non applicable",
        key="groupe_produit"
    )
    type_compte = st.selectbox(
        "Type de compte",
        options=sorted(df['TYPE_COMPTE'].dropna().unique()),
        help="Type du compte bancaire",
        key="type_compte"
    )
    # CATEGORIE_COMPTE est déduite automatiquement depuis TYPE_COMPTE
    categorie_compte = default_categorie.get(type_compte, 1001.0)
    famille_produit = st.selectbox(
        "Famille produit",
        options=sorted(df['FAMILLE_PRODUIT'].dropna().unique()),
        disabled=(nb == 0),
        help="Famille de produits" if nb > 0 else "Aucun produit — non applicable",
        key="famille_produit"
    )
    dossier_complet = st.selectbox(
        "Dossier complet",
        options=["YES", "NO"],
        help="Le dossier client est-il complet ?",
        key="dossier_complet"
    )

with col2:
    nb_produits = st.number_input(
        "Nombre de produits souscrits",
        min_value=0, max_value=50, value=1, step=1,
        on_change=on_nb_produits_change,
        key="nb_produits"
    )
    solde_compte = st.number_input(
        "Solde du compte (TND)",
        min_value=-1000000.0, value=1000.0, step=100.0,
        format="%.2f",
        key="solde_compte"
    )
    montant_produit = st.number_input(
            "Montant du produit (TND)",
            min_value=0.0, value=0.0 if nb == 0 else 5000.0, step=100.0,
            format="%.2f",
            disabled=(nb == 0),
            key="montant_produit"
        )
    anciennete_client = st.number_input(
        "Ancienneté client (années)",
        min_value=0.0, max_value=80.0, value=5.0, step=0.5,
        format="%.1f",
        key="anciennete_client"
    )
    age = st.number_input(
        "Âge du client",
        min_value=18, max_value=120, value=35, step=1,
        key="age"
    )

st.markdown("---")
st.button("Prédire le risque de churn", on_click=submit, use_container_width=True)

# ---- Prédiction ----
if st.session_state.submitted:
    st.session_state.submitted = False  # reset
    # 1. Construire le DataFrame client
    est_sans_produit = (nb_produits == 0)
    client = pd.DataFrame([{
        'GROUPE_PRODUIT': np.nan if est_sans_produit else groupe_produit,
        'DOSSIER_COMPLET': dossier_complet,
        'TYPE_COMPTE': type_compte,
        'CATEGORIE_COMPTE': categorie_compte,
        'NB_PRODUITS': nb_produits,
        'SOLDE_COMPTE': solde_compte,
        'MONTANT_PRODUIT': 0.0 if est_sans_produit else montant_produit,
        'ANCIENNETE_CLIENT': anciennete_client,
        'AGE': age,
        'FAMILLE_PRODUIT': "N/A"
    }])

    # 2. Label Encoding (DOSSIER_COMPLET)
    client['DOSSIER_COMPLET'] = client['DOSSIER_COMPLET'].map(le_map).fillna(0)

    # 3. One-Hot Encoding (FAMILLE_PRODUIT)
    client = pd.get_dummies(client, columns=['FAMILLE_PRODUIT'], drop_first=True)

    # 4. Target Encoding
    for col in ['GROUPE_PRODUIT', 'TYPE_COMPTE', 'CATEGORIE_COMPTE']:
        client[col] = client[col].map(encoding_maps[col]).fillna(encoding_maps[col + '_global_mean'])

    # 5. Alignement des colonnes
    for col in feat_cols:
        if col not in client.columns:
            client[col] = 0.0
    client = client[feat_cols]

    # 6. StandardScaler (appliqué seulement aux 8 colonnes numériques + target-encodées)
    client[scaler_cols] = scaler.transform(client[scaler_cols])

    # 7. Prédiction
    proba = model.predict_proba(client)[0, 1]
    seuil = 0.5
    prediction = 1 if proba >= seuil else 0

    # 8. Niveau de risque
    if proba < 0.30:
        risque = "🟢 Faible"
    elif proba < 0.60:
        risque = "🟡 Moyen"
    else:
        risque = "🔴 Élevé"

    # ---- Affichage des résultats ----
    st.markdown("---")
    st.subheader("Résultat de la prédiction")

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.metric(
            label="Probabilité de churn",
            value=f"{proba:.1%}"
        )

    with col_res2:
        decision = "CHURN" if prediction == 1 else "NON CHURN"
        st.metric(
            label="Décision",
            value=decision
        )

    with col_res3:
        st.metric(
            label="Niveau de risque",
            value=risque
        )

    # Barre de progression
    st.progress(proba, text=f"Risque de churn : {proba:.1%}")

    # Message contextualisé
    if proba < 0.30:
        st.success("🟢 Client stable — aucune action urgente requise.")
    elif proba < 0.60:
        st.warning("🟡 Surveillance rapprochée recommandée.")
    else:
        st.error("🔴 Action de rétention immédiate requise !")

    # Facteurs de risque
    st.markdown("---")
    st.subheader("Caractéristiques saisies")

    aff_groupe = "Aucun" if est_sans_produit else groupe_produit
    aff_famille = "N/A" if est_sans_produit else famille_produit
    aff_montant = "0 TND" if est_sans_produit else f"{montant_produit:,.2f} TND"
    features_df = pd.DataFrame({
        'Caractéristique': [
            'Groupe produit', 'Type de compte',
            'Famille produit', 'Dossier complet',
            'Nb produits', 'Solde compte', 'Montant produit',
            'Ancienneté', 'Âge'
        ],
        'Valeur': [
            aff_groupe, type_compte,
            aff_famille, dossier_complet,
            nb_produits, f"{solde_compte:,.2f} TND", aff_montant,
            f"{anciennete_client} ans", f"{age} ans"
        ]
    })
    st.dataframe(features_df, hide_index=True, use_container_width=True)
