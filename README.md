# Projet de Maîtrise de la Vision par Ordinateur (CV Mastery)

Ce projet est une application web éducative et interactive construite avec **Streamlit** et **OpenCV**. Il guide l'utilisateur à travers les étapes fondamentales de la vision par ordinateur, du traitement de base des pixels jusqu'à la classification d'images par Deep Learning.

## Fonctionnalités

L'application est divisée en quatre modules principaux :

### 1. Preprocessing (Nettoyer)
Cette étape permet de préparer l'image en réduisant le bruit et en améliorant la qualité visuelle.
-   **Conversion d'espace colorimétrique** : HSV, YUV, Grayscale.
-   **Égalisation d'histogramme (CLAHE)** : Amélioration du contraste local.
-   **Floutage (Blur)** : Gaussien (lissant) et Médian (suppression du bruit poivre et sel).
-   **Filtres de contours** : Sobel et Laplacien.

### 2. Segmentation (Isoler)
Cette étape vise à séparer les objets d'intérêt de l'arrière-plan.
-   **Seuillage (Thresholding)** : Simple et Otsu (automatique).
-   **Clustering** : K-Means et GMM (Gaussian Mixture Models) pour segmenter par couleur.
-   **Watershed** : Algorithme de séparation basé sur la topographie de l'image.

### 3. Analyse Classique (Mesurer)
Extraction d'informations mesurables à partir de l'image traitée.
-   **Détection de contours (Canny)** : Identification des bords des objets.
-   **Extraction de caractéristiques** : Calcul de l'aire, du périmètre, de la circularité et de l'aspect ratio des objets détectés.
-   **Histogramme de couleur** : Visualisation de la répartition des intensités RGB.

### 4. Deep Learning (Classifier)
Utilisation de réseaux de neurones profonds pour classifier l'image entière.
-   **Architectures supportées** : ResNet50, MobileNetV2, InceptionV3, Vision Transformer (ViT).
-   **Modes** : 
    -   *ImageNet* (1000 classes génériques).
    -   *CIFAR-10* (Modèle spécialisé pour animaux/véhicules).
    -   *Modèles personnalisés* : Possibilité de charger vos propres poids `.h5` ou `.keras`.

## Structure du Projet

```
CV/
├── app.py              # Point d'entrée de l'application Streamlit
├── assets/             # Fichiers statiques (CSS, images...)
│   └── style.css       # Styles personnalisés pour l'interface
├── models/             # Dossier pour les modèles entraînés (ex: MobileNet CIFAR-10)
├── notebooks/          # Notebooks Jupyter pour l'entraînement et l'exploration
├── src/                # Code source des modules
│   ├── analysis.py     # Fonctions d'analyse (contours, features)
│   ├── classifier.py   # Gestion des modèles de Deep Learning
│   ├── preprocessing.py # Fonctions de traitement d'image
│   └── segmentation.py # Algorithmes de segmentation
└── requirements.txt    # Liste des dépendances Python
```

## Installation

1.  **Cloner le dépôt** (si applicable) ou télécharger les fichiers.
2.  **Créer un environnement virtuel** (recommandé) :
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```
3.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

Pour lancer l'application, exécutez la commande suivante depuis la racine du projet :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut.

---

## 👨‍💻 Auteur

Ce projet a été développé par **Mohamed ZAHZOUH**.

- 🌍 **LinkedIn** : [Mohamed ZAHZOUH](https://www.linkedin.com/in/mohamed-zahzouh-1402a7318/)
- 📧 **Contact** : [mohamedzahzouh2006@gmail.com](mailto:mohamedzahzouh2006@gmail.com)

---

<center>
  <sub>Réalisé avec ❤️ et Python.</sub>
</center>

