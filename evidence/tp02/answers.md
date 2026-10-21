# TP02 – Answers and evidence / Réponses et preuves

Commit with the training code: `cbd7081` (branch `feat/tp02-training-pipeline`, tag `tp02-model`).
Pull request: https://github.com/Chanezig/ML-tp1/pull/2, merged into `main` as `c74a0ab`.
Artifact submission: GitHub Release https://github.com/Chanezig/ML-tp1/releases/tag/tp02-model, with `churn_pipeline.joblib` and `churn_pipeline.metadata.json`. I re-downloaded the file from the Release: same SHA-256, `0cde2a53…19080`.
Configuration: `configs/tp02.json` (seed 42, threshold 0.5, selection rule fixed before running).

## 1. First model / Premier modèle

Command: `uv run python scripts/first_model.py` → `first_model.json`

Positive class: `Churn = Yes` = 1. Confusion matrix layout: rows = true [No, Yes], columns = predicted [No, Yes]. Validation set: 1,409 customers (1,035 No / 374 Yes).

| Model (fitted on train, evaluated on validation) | Accuracy | Confusion matrix |
|---|---:|---|
| Majority baseline (`DummyClassifier(most_frequent)`) | 0.735 | `[[1035, 0], [374, 0]]` |
| Logistic regression (`tenure`, `MonthlyCharges`) | 0.783 | `[[950, 85], [221, 153]]` |

**FR**
- *Un système qui prédit toujours « pas de churn »* obtient 73,5 % d'accuracy, mais il ne trouve **aucun** des 374 clients qui partent (rappel = 0).
- *Une accuracy élevée peut donc aller avec 0 churner détecté*, à cause du déséquilibre des classes (26,5 % de Yes).
- *La baseline est entraînée sur le train*, comme les autres modèles. Elle apprend la classe majoritaire du train, puis elle est évaluée sur la validation.
- *Les données qui ont le droit d'influencer le choix du modèle* : le train sert à apprendre, la validation sert à comparer. Le test ne sert jamais à choisir.

**EN**
- *A system that always predicts "no churn"* reaches 73.5 % accuracy but finds **none** of the 374 churners (recall = 0).
- *High accuracy can coexist with zero detected churners* because of the class imbalance (26.5 % Yes).
- *The baseline is fitted on train*, like the other models: it learns the training majority class, then is evaluated on validation.
- *Data allowed to influence model selection*: train to learn, validation to compare. Test never influences the choice.

## 2. Preprocessing pipeline / Pipeline de prétraitement

Features: numeric `tenure`, `MonthlyCharges`, `TotalCharges`; categorical `Contract`, `InternetService`, `PaymentMethod`.

**Input parsing, outside the pipeline, stateless (`churnlab.data.prepare_features`)**
- The CSV is read as text. Numeric columns are converted explicitly: blank or whitespace-only → NaN. Any other non-numeric text raises `ValueError`. We do not use `errors="coerce"` alone, because it would silently turn `"abc"` into a missing value. We coerce, then reject every failure that was not blank.
- 11 blank `TotalCharges` in total: 7 in train, 1 in validation, 3 in test. The original CSV is never modified.

**Inside the pipeline, learned on train only**
- Numeric: `SimpleImputer(median)` → `StandardScaler`.
- Categorical: `SimpleImputer(constant="missing")` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Unknown category at inference: encoded as all zeros for that feature, so no crash. This does not mean the prediction is reliable for that customer.

**FR – À réfléchir**
- `fit` apprend les paramètres (médianes, moyennes, écarts-types, catégories, poids du modèle). `transform` applique ces paramètres appris aux données. `predict` transforme puis donne la classe (ou `predict_proba` la probabilité). Une fois le modèle entraîné, la validation et les nouvelles données n'appellent que `transform` et `predict`.
- *Ajuster un scaler sur la validation* ferait apprendre des statistiques sur des données qui doivent rester « inconnues » : c'est une fuite de données. En plus, la validation ne serait plus transformée comme le train, et les scores seraient trop optimistes.
- *Un Pipeline n'empêche pas la fuite tout seul.* Si on appelle `fit` sur le jeu complet, ou sur la validation, il apprend aussi sur ces données. Il protège seulement si on lui donne le bon jeu d'entraînement.
- *`Pipeline` ordonne les étapes* (prétraitement puis classifieur). *`ColumnTransformer` aiguille les colonnes* : numériques vers une branche, catégorielles vers l'autre.

**EN – Think about it**
- `fit` learns parameters (medians, means, scales, categories, model weights). `transform` applies these learned parameters. `predict` transforms then outputs the class (or `predict_proba` the probability). After training, validation and new data only go through `transform` and `predict`.
- *Fitting a scaler on validation* learns statistics from data that must stay unseen: that is leakage. Validation is then no longer transformed like train, and scores become optimistic.
- *A Pipeline does not prevent leakage by itself.* If you call `fit` on the full data or on validation, it learns from them too. It only protects you when it receives the correct training data.
- *`Pipeline` orders the steps* (preprocessing, then classifier). *`ColumnTransformer` routes the columns* (numeric to one branch, categorical to the other).

## 3. Fair comparison / Comparaison équitable

Same six features, same split (seed 42), same metrics, fresh pipeline for each candidate (`build_pipeline` clones the classifier). Command: `uv run python scripts/train.py` → `comparison.csv`, `comparison.json`.

**Selection metric, chosen before running:** validation **average precision** of the class-1 probability. It measures how well churners are ranked above non-churners, does not depend on a threshold, and is not inflated by the 73 % majority class (baseline AP = churn rate = 0.265).
**Tie rule:** if a simpler model is within 0.01 of the best AP, choose the simpler one (logistic regression < decision tree < random forest).

| Model | Config | AP | ROC AUC | Precision | Recall | F1 | Accuracy | TN / FP / FN / TP | Fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Majority baseline | most_frequent | 0.265 | 0.500 | 0.000 | 0.000 | 0.000 | 0.735 | 1035 / 0 / 374 / 0 | ~0.02 |
| Logistic regression | max_iter=1000 | 0.647 | 0.849 | 0.664 | 0.529 | 0.589 | 0.804 | 935 / 100 / 176 / 198 | ~0.05 |
| Decision tree | depth 5, leaf 20 | 0.617 | 0.838 | 0.636 | 0.602 | 0.618 | 0.803 | 906 / 129 / 149 / 225 | ~0.03 |
| Random forest | 300 trees, depth 8, leaf 10 | **0.664** | 0.853 | 0.653 | 0.484 | 0.556 | 0.795 | 939 / 96 / 193 / 181 | ~0.9 |

Threshold metrics at 0.5. Timings are approximate, on my laptop.

### Selection note / Note de sélection

**FR :** La random forest a la meilleure AP en validation (0,664). La régression logistique suit à 0,647 : l'écart (0,017) dépasse la tolérance de 0,01 fixée à l'avance, donc la règle de simplicité ne s'applique pas. **Modèle choisi : random forest.** La décision repose uniquement sur la validation.
À noter : avec le seuil 0,5, l'arbre de décision a le meilleur rappel (0,60) et le meilleur F1. Si l'équipe veut surtout *attraper* les churners, il faudra choisir un seuil plus bas sur la validation (extension 2), au lieu de changer de modèle.
On ne peut pas dire que l'algorithme s'est amélioré par rapport au premier modèle à 2 variables : les variables ont changé en même temps.

**EN:** Random forest has the best validation AP (0.664). Logistic regression follows at 0.647; the gap (0.017) is above the 0.01 tolerance fixed in advance, so the simplicity rule does not apply. **Selected model: random forest.** The decision uses validation only.
Note: at threshold 0.5, the decision tree has the best recall (0.60) and F1. If the team mainly wants to *catch* churners, a lower threshold should be chosen on validation (extension 2) rather than changing the model.
We cannot claim an algorithm improvement over the first 2-feature model: the feature set also changed.

**Held-out test, evaluated once after the choice (`test_evaluation.json`):** AP 0.635, ROC AUC 0.832, precision 0.615, recall 0.479, F1 0.538, accuracy 0.782. These results are close to validation, so the choice did not overfit the validation set much. They were not used to change anything.

**FR – À réfléchir**
- *Changer le split pour chaque modèle* rendrait la comparaison impossible à interpréter : une différence pourrait venir du découpage et non du modèle.
- *Le modèle avec la meilleure accuracy* n'est pas forcément le bon. Ici, la logistique a la meilleure accuracy, mais l'équipe veut repérer les churners, ce qui demande de bien les classer (AP) et d'en trouver une part suffisante (rappel).
- *Si deux modèles sont proches*, on prend le plus simple : plus rapide, plus facile à expliquer et à maintenir, moins de risque de sur-apprentissage. D'où la tolérance de 0,01.
- *Une probabilité de 0,8 n'est pas automatiquement un risque de 80 %.* Les modèles ne sont pas calibrés (une random forest a tendance à « tasser » ses probabilités). Il faudrait une courbe de calibration pour le vérifier.

**EN – Think about it**
- *Changing the split for each classifier* would make the comparison uninterpretable: differences could come from the split, not the model.
- *The highest-accuracy model* is not necessarily what the team needs. Here logistic regression has the best accuracy, but the team needs churners ranked well (AP) and found (recall).
- *If two candidates are close*, prefer the simpler one: faster, easier to explain and maintain, less overfitting risk. Hence the 0.01 tolerance.
- *A probability of 0.8 is not automatically an 80 % risk*: the models are not calibrated (random forests tend to compress probabilities). A calibration curve would be needed.

## 4. Reusable code and artifact / Code réutilisable et artefact

| Responsibility | Module / script |
|---|---|
| Load + verify hash + parse inputs | `src/churnlab/data.py` |
| Split by customer, allocation table | `src/churnlab/splits.py` |
| Build an **unfitted** pipeline | `src/churnlab/pipeline.py` |
| Candidate classifiers from config | `src/churnlab/models.py` |
| Metrics (positive class from `classes_`) | `src/churnlab/evaluation.py` |
| Code and environment identity | `src/churnlab/provenance.py` |
| Training entry point | `scripts/train.py` |
| Inference only | `scripts/predict.py` |
| Single test evaluation | `scripts/evaluate_test.py` |

- Artifact: `artifacts/churn_pipeline.joblib`, the fitted preprocessor and classifier in one object. SHA-256 `0cde2a53e7e0f56f32ad267d149c98a7690d57c2efd31ecb2f700d3ae0b19080`.
- Metadata: `model_metadata.json`, with source hash, seed, allocation hash, feature roles and input types, preprocessing settings, parameters, validation metrics, positive class, threshold, Git commit `cbd7081` (no uncommitted changes) and library versions.
- **Reload:** `scripts/predict.py` in a separate process, with no `fit` and no `Churn` column, on `sample_batch.csv`. Its probabilities match `original_predictions.csv` with a maximum difference of 1.7e-16 (tolerance 1e-12).
- **Determinism:** retraining from the commit gives the same artifact SHA-256 and the same metrics. Probabilities can differ around 1e-16 (multithreaded random forest summation), which justifies the tolerance.

**FR – À réfléchir**
- *Sauvegarder seulement le classifieur ne suffit pas* : il attend des données déjà imputées, standardisées et encodées. Sans le prétraitement ajusté, il faudrait refaire ces étapes à la main, avec un risque d'erreur ou de fuite.
- *Ce qui est appris dans l'imputer et l'encodeur* : les médianes du train (29 ; 70,55 ; 1 373,625), les moyennes et écarts-types du scaler, et la liste des catégories vues (Contract : 3 valeurs, InternetService : 3, PaymentMethod : 4).
- *Deux fichiers `model.joblib` peuvent contenir des modèles différents* : le nom ne prouve rien. On identifie l'artefact par son SHA-256, inscrit dans les métadonnées et vérifié par `predict.py`.
- *Pipeline sklearn ou DAG Kubeflow* : un Pipeline sklearn est **un objet** qui enchaîne des transformations et un modèle dans un même processus Python, et qu'on sauvegarde. Un DAG Kubeflow orchestre **des étapes séparées** (conteneurs : téléchargement, entraînement, évaluation, déploiement), avec leurs entrées/sorties, sur une infrastructure.

**EN – Think about it**
- *Saving only the classifier is insufficient*: it expects imputed, scaled and encoded inputs. Without the fitted preprocessing, these steps would have to be rebuilt by hand (risk of mismatch or leakage).
- *Fitted information in the imputer/encoder*: training medians (29; 70.55; 1373.625), scaler means and scales, and the categories seen (Contract 3, InternetService 3, PaymentMethod 4).
- *Two `model.joblib` files can contain different models*: a name proves nothing. The SHA-256 in the metadata identifies the artifact and `predict.py` checks it.
- *sklearn Pipeline vs Kubeflow DAG*: a sklearn Pipeline is **one object** chaining transforms and an estimator inside one Python process, saved as a whole. A Kubeflow DAG orchestrates **separate steps** (containers for download, training, evaluation, deployment) with their inputs and outputs on infrastructure.

## 5. Reliability checks / Vérifications de fiabilité

Command: `uv run pytest -v` → `checks_pytest.txt` (**9 passed**). Inspection of the selected artifact: `artifact_checks.json`.

| Check | Evidence |
|---|---|
| Customer IDs do not overlap; union = all 7,043 | `test_customer_ids_disjoint_and_complete` |
| Same seed → same allocation; churn rate equal across splits (±0.5 pt) | `test_split_is_reproducible_and_stratified` |
| Imputer medians = **train** medians, not full data | `artifact_checks.json`: learned `TotalCharges` median 1373.625 = train; full data = 1397.475; `MonthlyCharges` 70.55 vs 70.35 |
| `predict` does not refit | medians, scaler means and scales unchanged after predicting validation (`test_prediction_does_not_refit`, `medians_unchanged_after_predict: true`) |
| Blank numeric input → training median | `test_blank_numeric_is_imputed_with_training_median` |
| Unseen category (`Contract = "Five year"`) → all-zero encoding, still a score | `test_unseen_category_is_encoded_as_zeros` (no reliability claim for it) |
| Malformed number (`"abc"`) → visible `ValueError` | `test_malformed_numeric_text_fails_visibly` |
| Missing required column → visible `ValueError` (boundary **and** pipeline) | `test_missing_required_feature_fails_visibly` |
| Save/reload in a fresh process, no labels, no training | `test_reload_in_fresh_process_preserves_predictions` (atol 1e-12) + `scripts/predict.py` |

## Optional extension 3 – Failure experiment / Expérience de panne

Tests performed on copied inference batches:
- **Schema failure:** the `Contract` column is dropped, or `MonthlyCharges = "abc"`. The input does not match the contract, so we **stop with an error**. A wrong silent prediction would be worse.
- **Supported missingness:** `TotalCharges = " "`. It is planned for, documented and imputed with the training median. The prediction is produced normally.
- **Distribution shift:** `Contract = "Five year"`, or a customer population very different from training. Nothing crashes (`handle_unknown="ignore"`), but the model has never seen this case, so the score can be wrong without any warning. This requires **monitoring** of input distributions (later TP), not a parsing rule.

## Peer review / Relecture

**FR :** Pas de binôme disponible : auto-relecture depuis un **clone neuf** de `feat/tp02-training-pipeline`, en suivant uniquement le README.
- **Question :** Un autre ingénieur peut-il réentraîner et recharger l'artefact avec seulement le README ?
- **Résultat :** `uv sync --locked`, `download_data.py`, `first_model.py`, `train.py`, `pytest` (9 passed), `predict.py`, `evaluate_test.py` fonctionnent tous. L'artefact obtenu a **le même SHA-256** (`0cde2a53…`), et les métriques de validation et de test sont identiques.
- **Problème trouvé :** `train.py` réécrivait les fichiers de preuve versionnés (`evidence/tp02/comparison.*`, `model_metadata.json`, `original_predictions.csv`), à cause des temps d'entraînement, de la date et d'écarts d'environ 1e-16. L'arbre Git du relecteur devenait « modifié ».
- **Correction :** les sorties vont par défaut dans `reports/tp02/` (ignoré par Git). Les preuves ne sont rafraîchies qu'avec `--evidence-dir evidence/tp02`. README mis à jour. Re-test : l'arbre reste propre.

**EN:** No partner was available: self-review from a **fresh clone** of `feat/tp02-training-pipeline`, following only the README.
- **Question:** Can another engineer retrain and reload the artifact using only the README?
- **Result:** all commands work; the artifact has **the same SHA-256** (`0cde2a53…`) and identical validation/test metrics.
- **Issue found:** `train.py` overwrote committed evidence files (fit times, timestamp, ~1e-16 differences), leaving the reviewer's working tree modified.
- **Resolution:** outputs now default to the ignored `reports/tp02/`; evidence is refreshed only with `--evidence-dir evidence/tp02`. README updated. Re-tested: the working tree stays clean.

## Remaining limitations / Limites restantes

- Probabilities are not calibrated; threshold 0.5 is not tuned to a business cost.
- Small hyperparameter configurations, chosen by hand; no cross-validation.
- The artifact depends on the exact library versions (pickle); OS and hardware are not pinned (no container).
- No experiment tracking or model registry yet (next TP).
- The hash proves the data bytes, not the label quality.
