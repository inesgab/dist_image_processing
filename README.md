# Fluo Measuring Project

## Description
Ce projet permet de traiter des images et de générer des masques de goutteletes, ainsi que de les tracker dans le temps. Le tracking permet ensuite de mesurer la fluorescence dans chaque goulette.

## Installation

### Prérequis
- Python >= 3.10 (et < 3.13)
- Installation de sam2 (Meta, lien: https://github.com/facebookresearch/sam2/tree/main, checkpoint utilisé (à télécharger dans le repo dist_image_processing): `sam2.1_hiera_large.pt`, également dans le github sam2)
- Bibliothèques nécessaires (cf.  `requirements.txt`)

### Tracking des cellules

#### Le format des données d'entrée

Les données doivent être stockées dans le fichier `data` :
- `data/<nom_dossier>/<nom_dossier>_t???_EGFP_ORG.tif` pour les images fluo
- `data/<nom_dossier>/<nom_dossier>_t???_DIC II 40x_ORG.tif` pour les images DIC

avec `t???`représentant le numéro de frame (ex: `t001`). Pour l'instant, pas de step>1 possible. Le code est fait pour des timestamps à 3 digits pour l'instant.

#### Lancer le tracking
depuis `src`, exécuter `main.py` et remplir les infos nécessaires


#### Données de sortie (tracking)
A chaque dossier d'entrée correspond un dossier de sortie dans le dossier `results/`. Dans `results/<nom_dossier>` se trouvent des dossiers correspondant chacun à une goutte. Dans `results/<nom_dossier>/1/`, on trouvera trois dossiers `fluo/`, `mask/` et `overlay/`(qui permet de vérifier si chaque masque correspond bien aux contours d'une goutte).


#### Plotting fluorescence
Les dossiers `fluo`et `mask`servent à lancer un deuxième script. Depuis src, exécuter `run_fluo.py`.

Le tableau résultat et la figure de l'évolution de la fluorescence par goutte en fonction du temps se trouvent dans `results/<nom_dossier>/`sous le nom de `fluorescence_results.csv`et `fluorescence_plot.png` respectivement.