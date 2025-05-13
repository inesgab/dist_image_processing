# Fluo Measuring Project

## Description
Ce projet permet de traiter des images et de générer des masques de goutteletes, ainsi que de les tracker dans le temps. Le tracking permet(tra) ensuite de mesurer la fluorescence dans chaque goulette.

## Installation

### Prérequis
- Python >= 3.10 (et < 3.13)
- Installation de sam2 (Meta)
- Bibliothèques nécessaires (cf.  `requirements.txt`)

### Tracking des cellules

#### Le format des données d'entrée

Les données doivent être stockées dans le fichier `data` :
- `<nom_dossier>/<nom_dossier>_t???_EGFP_ORG.tif` pour les images fluo
- `<nom_dossier>/<nom_dossier>_t???_DIC II 40x_ORG.tif` pour les images DIC

avec `t???`représentant le numéro de frame (ex: `t001`). Pour l'instant, pas de step>1 possible (codable si besoin).

#### Données de sortie (tracking)
A chaque dossier d'entrée correspon un dossier de sortie dans le dossier `results/`. Dans `results/<nom_dossier>` se trouvent des dossiers correspondant chacun à une goutte, et à sa roi fluo + masque par frame. 