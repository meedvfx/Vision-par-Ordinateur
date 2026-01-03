# 👁️ Vision par Ordinateur

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

> **Une application interactive pour explorer la Vision par Ordinateur, du traitement de pixel au Deep Learning.**

Ce projet est une application web éducative construite avec **Streamlit** et **OpenCV**. Il guide l'utilisateur à travers les étapes fondamentales de la vision par ordinateur de manière intuitive et visuelle.

---

## ✨ Fonctionnalités

L'application est divisée en quatre modules principaux, chacun explorant une facette de la vision par ordinateur :

### 1. 🧹 Preprocessing (Nettoyer)
*Préparez vos images pour l'analyse en améliorant leur qualité.*
*   🎨 **Conversion d'espace colorimétrique** : HSV, YUV, Grayscale.
*   ⚖️ **Égalisation d'histogramme (CLAHE)** : Révélez les détails cachés.
*   🌫️ **Floutage (Blur)** : Gaussien pour lisser, Médian pour débruiter.
*   ✏️ **Filtres de contours** : Sobel et Laplacien pour détecter les gradients.

### 2. 🧩 Segmentation (Isoler)
*Séparez les objets d'intérêt du fond.*
*   ⚫⚪ **Seuillage (Thresholding)** : Simple ou Otsu (automatique).
*   🌈 **Clustering** : K-Means et GMM pour une segmentation basée sur la couleur.
*   🏞️ **Watershed** : Segmentation topographique avancée.

### 3. 📏 Analyse Classique (Mesurer)
*Extrayez des données quantitatives de vos images.*
*   📐 **Détection de contours (Canny)** : Trouvez les limites précises des objets.
*   📊 **Extraction de caractéristiques** : Aire, périmètre, circularité, ratio.
*   📈 **Histogramme de couleur** : Analysez la répartition spectrale RGB.

### 4. 🧠 Deep Learning (Classifier)
*Exploitez la puissance de l'IA pour reconnaître le contenu.*
*   🏗️ **Architectures SOTA** : ResNet50, MobileNetV2, InceptionV3, Vision Transformer (ViT).
*   🎯 **Modes** : 
    *   *ImageNet* (1000 classes).
    *   *CIFAR-10* (Spécialisé animaux/véhicules).
    *   *Custom* : Chargez vos propres modèles `.h5` / `.keras`.

---

## 📂 Structure du Projet

Une organisation claire pour un développement sain :

```bash
CV/
├── app.py              # 🚀 Point d'entrée de l'application
├── assets/             # 🎨 Ressources statiques (Styles, Images)
├── models/             # 🤖 Modèles de Deep Learning
├── notebooks/          # 📓 Expérimentations Jupyter
├── src/                # 🧱 Code source modulaire
│   ├── analysis.py     # Logique d'analyse
│   ├── classifier.py   # Moteur d'inférence IA
│   ├── preprocessing.py # Algorithmes de traitement
│   └── segmentation.py # Algorithmes de segmentation
├── requirements.txt    # 📦 Dépendances
└── README.md           # 📖 Documentation
```

---

## ⚙️ Installation

Configurez votre environnement en quelques secondes :

1.  **Cloner le projet** :
    ```bash
    git clone https://github.com/votre-username/CV.git
    cd CV
    ```

2.  **Créer un environnement virtuel** (Recommandé) :
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate
    
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Utilisation

Lancez l'interface web avec une simple commande :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur ! 🎉

---

## 👨‍💻 Auteur

Ce projet a été développé par **Mohamed ZAHZOUH**.

- 🌍 **LinkedIn** : [Mohamed ZAHZOUH](https://www.linkedin.com/in/mohamed-zahzouh-1402a7318/)
- 📧 **Contact** : [mohamedzahzouh2006@gmail.com](mailto:mohamedzahzouh2006@gmail.com)

---

<center>
  <sub>Réalisé avec ❤️ et Python.</sub>
</center>
