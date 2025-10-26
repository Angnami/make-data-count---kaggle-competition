# 🧠 Compétition Kaggle : Make Data Count - Finding Data References

## 🚀 Vue d'ensemble du Projet

Ce dépôt contient le code et l'approche détaillée soumis à la compétition Kaggle **"Make Data Count - Finding Data References"**.  
L'objectif était d'identifier, d'extraire et de classer les mentions de jeux de données (*Data References*) au sein d'articles scientifiques, distinguant leur type d'utilisation :  

- **Primaire** (génération du jeu de données);
- **Secondaire** (réutilisation de données existantes).  

Notre solution repose sur un **pipeline d'apprentissage automatique en deux étapes** (**NER + Classification**) combiné à des techniques de **post-traitement heuristique** afin de garantir la robustesse et la précision des identifiants extraits (DOI et Accession IDs).

---

## 💡 Approche Technique

Le pipeline de solution est structuré en plusieurs étapes critiques :

### 1. Pré-traitement et Nettoyage des Données (Data Cleaning)

**Extraction de Texte Robuste :**  
Utilisation de **PyMuPDF (fitz)** pour extraire le texte brut à partir des fichiers PDF, avec une logique de nettoyage pour supprimer les en-têtes et pieds de page répétitifs.  
Le texte des fichiers XML a été extrait à l’aide de `xml.etree.ElementTree`.

**Nettoyage des Labels d’Entraînement :**  
Correction manuelle et automatique des mentions de jeux de données dans `train_labels.csv` qui étaient tronquées, mal formatées ou incohérentes avec le texte source (≈ 50 cas identifiés et corrigés).

**Standardisation des DOI :**  
Normalisation de tous les identifiants vers un format uniforme :  
`https://doi.org/10.xxxx/...`

---

### 2. Modélisation : Extraction d’Entités Nommées (NER)

La première étape consiste à localiser toutes les mentions de DOI et d’Accession IDs dans le texte.

- **Modèle :** `allenai/scibert_scivocab_uncased` (fine-tuné)  
- **Tâche :** Token Classification (étiquetage BIO : `B-DOI`, `I-DOI`, `B-ACC`, `I-ACC`)  
- **Technique :** Fenêtre glissante (*Sliding Window*) avec chevauchement pour gérer les articles longs (max 512 tokens)  
- **Évaluation :** Le modèle NER a atteint un F1-score élevé sur l’ensemble de validation.

---

### 3. Post-traitement et Filtrage Heuristique

Après l'extraction NER, un ensemble de règles a été appliqué pour filtrer les faux positifs et améliorer la qualité :

- **Filtrage des Faux Positifs :**  
  Exclusion des mentions liées aux DOIs d’articles scientifiques ou à des numéros de financement, via des regex et mots-clés contextuels (`funding`, `grant`, etc.).  
- **Filtrage des Mentions Tronquées/Invalides :**  
  Suppression des identifiants trop courts ou ne respectant pas les formats DOI/Accession.  
- **Expansion des Accession IDs :**  
  Gestion des plages d’identifiants (ex : `GSE12345-GSE12350`) pour identifier automatiquement les entrées associées.

---

### 4. Modélisation : Classification de l’Usage (Primaire vs Secondaire)

La seconde étape consiste à **classer le type d’usage** des jeux de données extraits.

- **Modèle :** `allenai/scibert_scivocab_uncased` (fine-tuné)  
- **Tâche :** Sequence Classification (`Primary` ou `Secondary`)  
- **Contexte Enrichi :** Extension du contexte textuel au paragraphe complet autour de la mention, en particulier les sections *Data Availability*.  
- **Stratégie de Split :** `StratifiedGroupKFold` garantissant qu’un même article ne figure pas à la fois dans l’entraînement et la validation (prévention du data leakage).

---

## 🛠️ Dépendances

Le projet a été développé en **Python 3.x** et nécessite les bibliothèques principales suivantes :

| Bibliothèque | Description |
|---------------|-------------|
| `transformers` | Modèles SciBERT pour NER et Classification |
| `datasets` | Gestion des jeux de données Hugging Face |
| `scikit-learn` | Métriques et validation croisée (GroupShuffleSplit / StratifiedGroupKFold) |
| `pandas`, `numpy` | Manipulation et préparation des données |
| `pymupdf` (fitz) | Extraction de texte à partir des PDF |
| `seqeval` | Métriques NER (F1-score, précision, rappel) |
| `torch` | Backend d'entraînement (CUDA/GPU requis) |

---

## 📂 Structure du Code

Le code complet est contenu dans le notebook :  
`final-notebook-for-mdc-challenge.ipynb`

**Classes principales :**

| Classe | Description |
|---------|-------------|
| `ExtractTextFromArticles` | Extraction de texte propre à partir des fichiers PDF/XML |
| `DOIsFormatHandler` | Nettoyage, normalisation et validation des DOIs extraits |
| `TrainingDataPreparator` | Préparation et étiquetage BIO pour NER + contexte pour classification |
| `DatasetMentionFilter` | Application des règles heuristiques de filtrage et validation |
| `MentionExtractorAndClassifier` | Pipeline d’inférence combinée NER + classification |
| `FindDataReferences` | Pipeline complet orchestrant extraction, classification et génération de soumission |

---

## 📈 Résultats et Performances

### Modèle NER  

(F1-score micro sur l’ensemble de validation, hors classe `O`)

| Entité | Précision | Rappel | F1-Score |
|---------|------------|---------|----------|
| DOI | 0.91 | 0.88 | 0.90 |
| ACC | 0.93 | 0.91 | 0.92 |
| **Global** | **0.92** | **0.89** | **0.91** |

### Modèle de Classification  

(F1-score pondéré sur l’ensemble de validation)

| Métrique | Valeur |
|-----------|---------|
| Accuracy | 0.87 |
| F1-Score (Weighted) | 0.87 |

---

## 🔗 Lien vers la Compétition

👉 [Make Data Count - Finding Data References (Kaggle)](https://www.kaggle.com/competitions/make-data-count-finding-data-references)
