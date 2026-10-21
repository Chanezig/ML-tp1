Question A1

Why should the original CSV, a fitted model and the training source code have different roles in the repository?

English
The original CSV is the raw input data it must stay unchanged and be identifiable (source and SHA-256 hash). The training source code is what we write, review and version with Git. The fitted model is a generated result: it is produced by running the code on the data, so it can be rebuilt and should not be treated as source code. Keeping the three separate lets us know exactly which data and which code produced which model, and lets someone else reproduce the work.

Français
Le CSV d’origine est la donnée brute : il doit rester inchangé et identifiable (source et hash SHA-256). Le code d’entraînement est ce qu’on écrit, relit et versionne avec Git. Le modèle entraîné est un résultat généré : il est produit en exécutant le code sur les données, donc on peut le refaire et on ne le traite pas comme du code source. Garder ces trois éléments séparés permet de savoir exactement quelles données et quel code ont produit quel modèle, et à une autre personne de reproduire le travail.

Question A2

Why would including Churn among the model inputs invalidate the prediction task?

English
Churn is the target, the answer we want to predict. If it is given to the model as an input, the model simply reads the answer instead of learning from customer characteristics. This is target leakage. The scores would look excellent, but the model would be useless in practice, because we do not know whether a customer will churn at the time we want to predict it.

Français
Churn est la cible, c’est-à-dire la réponse qu’on veut prédire. Si on la donne au modèle comme entrée, il lit simplement la réponse au lieu d’apprendre à partir des caractéristiques du client. C’est une fuite de la cible (target leakage). Les scores paraîtraient excellents, mais le modèle serait inutilisable en pratique, car on ne sait pas si un client va partir au moment où on veut le prédire.


## B2

Francais

1. **Contrainte Python et dépendance directe** : `requires-python = ">=3.11"`. Dépendance directe : `pandas`, déclarée avec `pandas>=3.0.5` dans `pyproject.toml`.
2. **Dans `uv.lock`** : `pandas` apparaît avec la version exacte `3.0.5` (voir la remarque sur Smart App Control en partie D).
3. **Dépendance transitive** : `python-dateutil` 2.9.0.post0, installée parce que `pandas` en a besoin (je ne l'ai pas demandée directement).
4. **Différence entre les trois** :
   - `pyproject.toml` : ce que je demande (nom du projet, version de Python, dépendances avec des contraintes de versions).
   - `uv.lock` : les versions exactes résolues de tous les paquets, directs et transitifs.
   - `.venv/` : l'environnement installé sur ma machine, généré à partir des deux fichiers précédents. On ne le commite pas et on peut le recréer.

Remarque : mes contraintes sont des bornes basses (`>=`), donc seul `uv.lock` garantit que tout le monde installe les mêmes versions. Les versions viennent de mon propre `uv.lock`, pas de celui du projet de l'enseignant.

## B3

- `numpy ~= 1.26.0` signifie `>=1.26.0, <1.27.0` : toute version 1.26.x est autorisée.
- `numpy==1.26.4` fixe une seule version exacte.
- Un intervalle autorise plusieurs versions candidates ; le lock enregistre la version choisie, donc tout le monde installe la même.
- La numérotation seule ne garantit pas la compatibilité : une mise à jour mineure peut quand même casser quelque chose.

ENGLISH 
## B2

1. **Python constraint and direct dependency**: `requires-python = ">=3.11"`. Direct dependency: `pandas`, declared as `pandas>=3.0.5` in `pyproject.toml`.
2. **In `uv.lock`**: `pandas` appears with the exact version `3.0.5` (see the Smart App Control note in Part D).
3. **Transitive dependency**: `python-dateutil` 2.9.0.post0, installed because `pandas` requires it (I did not request it directly).
4. **Difference between the three**:
   - `pyproject.toml`: what I ask for (project name, Python version, dependencies with version constraints).
   - `uv.lock`: the exact resolved versions of all packages, direct and transitive.
   - `.venv/`: the environment installed on my machine, generated from the two previous files. It is not committed and can be recreated.

Note: my constraints are lower bounds (`>=`), so only `uv.lock` guarantees that everyone installs the same versions. The versions come from my own `uv.lock`, not from the instructor's project.

## B3

- `numpy ~= 1.26.0` means `>=1.26.0, <1.27.0`: any 1.26.x version is allowed.
- `numpy==1.26.4` pins a single exact version.
- A range allows several candidate versions; the lock file records the chosen one, so everyone installs the same version.
- Version numbering alone does not guarantee compatibility: even a minor update can still break something.


## C1 – C2 : Observations

Commande / Command : `uv run python scripts/inspect_telco.py` (rapport copié dans / report copied to `evidence/tp01/data-overview.json`).

| Observation | Attendu / Expected | Obtenu / Obtained |
|---|---:|---:|
| SHA-256 | `16320c9c…3055e91` | `16320c9c…3055e91` ✅ |
| Rows | 7 043 | 7 043 ✅ |
| Columns | 21 | 21 ✅ |
| `Churn = No` | 5 174 | 5 174 ✅ |
| `Churn = Yes` | 1 869 | 1 869 ✅ |
| Churn proportion | ≈ 26.54 % | 26.537 % ✅ |
| Duplicate customer IDs | 0 | 0 ✅ |
| Whitespace-only `TotalCharges` | 11 | 11 ✅ |

Source : `data/raw/telco.source.json` (dépôt IBM, commit `1fd6fd70906479be1712e64cfc0ea89f58466629`), produit par / produced by `scripts/download_data.py`.

## C3

Français

1. **Classification** : la cible `Churn` ne prend que deux valeurs (`Yes` / `No`). On prédit une catégorie, pas une quantité continue : c'est une classification binaire.
2. **Déséquilibre des classes** : environ 73,5 % de `No` contre 26,5 % de `Yes`. Un modèle qui répond toujours « No » obtient déjà ≈ 73,5 % d'accuracy sans détecter aucun client qui part. Il faudra donc regarder aussi le rappel, la précision, le F1 ou l'AUC sur la classe `Yes`.
3. **`TotalCharges` en texte** : 11 lignes contiennent un espace au lieu d'un nombre (clients avec `tenure = 0`). Comme une colonne CSV n'a qu'un seul type, pandas ne peut pas la convertir en `float` et la lit comme texte (`object`, ou `str` avec pandas 3, comme dans mon rapport).
4. **`isna()` ne suffit pas** : `" "` est une chaîne non vide, pas une valeur manquante (`NaN`). `isna()` renvoie donc `False`. Il faut d'abord nettoyer (`str.strip()`) puis comparer à `""`, ou convertir avec `pd.to_numeric(..., errors="coerce")`.
5. **Preuve du même fichier** : le hash SHA-256 (`16320c9c…3055e91`), identique chez les deux coéquipiers, plus l'URL épinglée sur un commit IBM dans `telco.source.json`. Le nombre de lignes/colonnes et les comptages de la cible confirment, mais seul le hash identifie les octets exacts.

English

1. **Classification**: the target `Churn` has only two values (`Yes` / `No`). We predict a category, not a continuous quantity: it is binary classification.
2. **Class imbalance**: about 73.5 % `No` vs 26.5 % `Yes`. A model that always answers "No" already reaches ≈ 73.5 % accuracy while detecting no churner. We will need recall, precision, F1 or AUC for the `Yes` class as well.
3. **`TotalCharges` as text**: 11 rows contain a space instead of a number (customers with `tenure = 0`). Since a CSV column gets a single type, pandas cannot parse it as `float` and reads it as text (`object`, or `str` with pandas 3 as in my report).
4. **`isna()` is not enough**: `" "` is a non-empty string, not a missing value (`NaN`), so `isna()` returns `False`. We must strip it and compare with `""`, or convert with `pd.to_numeric(..., errors="coerce")`.
5. **Evidence of the same file**: the SHA-256 hash (`16320c9c…3055e91`) matching on both machines, plus the commit-pinned IBM URL in `telco.source.json`. Row/column counts and target counts confirm it, but only the hash identifies the exact bytes.

## D1 – Recréation de l'environnement / Environment recreation

Étapes réalisées / Steps performed:

1. `evidence/tp01/environment.json` copié en / copied to `environment-before.json`.
2. `.venv/` renommé en / renamed to `.venv-backup/`.
3. `uv sync --locked` → nouvel environnement, 12 paquets installés depuis `uv.lock` / new environment, 12 packages installed from `uv.lock`.
4. `uv run python scripts/record_environment.py` et / and `uv run python scripts/inspect_telco.py`.
5. Comparaison / Comparison : `environment-before.json` et `environment.json` **identiques / identical** (Python 3.11.9, uv 0.13.4, churnlab 0.1.0, pandas 3.0.5, numpy 2.4.6, scikit-learn 1.9.1) ; rapport de données identique / identical data report ; hash du CSV identique / same CSV hash ; `uv.lock` non modifié / unchanged.
6. `.venv-backup/` supprimé / removed.

**Remarque – Smart App Control / Note – Smart App Control**

FR : Lors du premier essai, `uv sync --locked` réussissait mais `import pandas` échouait (`DLL load failed … Une stratégie de contrôle d'application a bloqué ce fichier`). Smart App Control de Windows 11 bloquait la DLL de `pandas 3.0.6`, publiée trop récemment pour avoir une « réputation ». J'ai abaissé la contrainte à `pandas>=3.0.5` et verrouillé `pandas 3.0.5` (`uv lock --upgrade-package pandas==3.0.5`), qui est accepté. Leçon : un lock file identique ne garantit pas un environnement utilisable ; la politique de sécurité de l'OS fait aussi partie de l'environnement.

EN: On the first attempt, `uv sync --locked` succeeded but `import pandas` failed (`DLL load failed … an application control policy blocked this file`). Windows 11 Smart App Control blocked the `pandas 3.0.6` DLL, released too recently to have a reputation. I lowered the constraint to `pandas>=3.0.5` and locked `pandas 3.0.5` (`uv lock --upgrade-package pandas==3.0.5`), which is allowed. Lesson: an identical lock file does not guarantee a working environment; the OS security policy is also part of the environment.

## D2

| Item | Evidence produced in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit (hash below) | Uncommitted edits are not identified by that commit |
| Package selections | `pyproject.toml`, `uv.lock` | Platform/interpreter selection can affect installations (e.g. Windows Smart App Control blocked `pandas 3.0.6`; numpy 2.5.4 is locked for Python ≥ 3.12 vs 2.4.6 for 3.11) |
| Installed environment | `environment-before.json` / `environment.json` | Does not capture every OS or hardware detail (CPU, BLAS, drivers, security policies) |
| Source data | Source URL, IBM commit and SHA-256 in `telco.source.json` | Does not establish label quality or absence of leakage |
| Trained model | Not produced in TP01 | Requires later training/configuration/artifact evidence (seed, hyperparameters, splits, metrics) |

## E – Git workflow

- Repository: https://github.com/Chanezig/ML-tp1
- Pull request: https://github.com/Chanezig/ML-tp1/pull/1
- Initial commit: `038e73e` (Initialize Telco churn project)
- PR commits: `a8eb281` (README), `4029149` (fix after review)
- Merge commit on `main`: `1f0aac5` (Merge pull request #1)

### Relecture / Review

FR : Pas de binôme disponible : j'ai fait une **auto-relecture** (GitHub ne permet pas d'approuver sa propre PR, la relecture est donc un commentaire sur la PR).

- **Question :** Puis-je reproduire la vérification des données depuis un environnement neuf en suivant uniquement le README ?
- **Vérification :** clone neuf de `docs/telco-onboarding` dans un dossier vide, puis `uv sync --locked`, `download_data.py` et `inspect_telco.py`. `uv.lock` inchangé, `churnlab 0.1.0`, CSV retéléchargé depuis le commit IBM épinglé, SHA-256 correct, comptages identiques (7 043 / 21 / 5 174 / 1 869 / 0 / 11).
- **Problème trouvé :** `download_data.py` réécrivait `recorded_at` à chaque exécution, donc `data/raw/telco.source.json` (versionné) apparaissait modifié après avoir suivi le README.
- **Correction :** commit `4029149`, la métadonnée n'est réécrite que si le hash ou l'URL change. Re-testé depuis le clone neuf : l'arbre de travail reste propre. PR mergée ensuite.

EN: No partner was available, so I did a **self-review** (GitHub does not allow approving your own PR, so the review is a PR comment).

- **Question:** Can I reproduce the dataset check from a fresh environment by following only the README?
- **Check:** fresh clone of `docs/telco-onboarding` in an empty folder, then `uv sync --locked`, `download_data.py`, `inspect_telco.py`. `uv.lock` unchanged, `churnlab 0.1.0`, CSV re-downloaded from the pinned IBM commit, SHA-256 correct, same counts (7,043 / 21 / 5,174 / 1,869 / 0 / 11).
- **Issue found:** `download_data.py` rewrote `recorded_at` on every run, so the committed `data/raw/telco.source.json` showed as modified after following the README.
- **Fix:** commit `4029149` rewrites the metadata only if the hash or URL changes. Re-tested from the fresh clone: the working tree stays clean. PR merged afterwards.

## F – Restore vs unstage

Français

- **F1 – `git restore README.md`** : remplace le fichier de travail par la version de l'index. La modification non indexée est **supprimée définitivement** (Git ne l'avait jamais enregistrée, on ne peut pas la récupérer).
- **F2 – `git restore --staged README.md`** : remplace seulement l'entrée de l'index par la version de `HEAD`. La modification **n'est plus indexée mais reste dans le fichier** : `git status --short` affiche ` M README.md` et `git diff` la montre toujours.
- En résumé : F1 agit sur le *working tree* (perte du travail), F2 agit sur l'*index* (on « désindexe » sans rien perdre).

English

- **F1 – `git restore README.md`**: overwrites the working file with the index version. The unstaged edit is **permanently discarded** (Git never stored it, so it cannot be recovered).
- **F2 – `git restore --staged README.md`**: only resets the index entry to the `HEAD` version. The edit is **no longer staged but stays in the file**: `git status --short` shows ` M README.md` and `git diff` still shows it.
- In short: F1 acts on the working tree (work is lost), F2 acts on the index (unstaging without losing anything).

## Ce que TP01 ne reproduit pas encore / What TP01 does not reproduce yet

- Aucun modèle entraîné : pas de seed, split, hyperparamètres ni métriques versionnés. / No trained model: no versioned seed, split, hyperparameters or metrics.
- Nettoyage de `TotalCharges` et contrat de données pas encore faits (TP02). / `TotalCharges` cleaning and data contract not done yet (TP02).
- OS, matériel et politiques de sécurité non figés (pas de conteneur). / OS, hardware and security policies not pinned (no container).
- Le hash prouve l'identité des octets, pas la qualité des labels. / The hash proves byte identity, not label quality.
- Le CSV brut n'est pas dans Git : on le récupère via `scripts/download_data.py` (URL épinglée + hash). / The raw CSV is not in Git: it is fetched with `scripts/download_data.py` (pinned URL + hash check).