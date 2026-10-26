"""
Application Streamlit : reconnaissance faciale (FaceNet + SVM).

Fichiers nécessaires dans le même dossier :
  - app.py
  - requirements.txt
  - embeddings.npz   (créé en Phase 2 : clés 'facenet' et 'labels')
"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC

st.set_page_config(page_title="Reconnaissance faciale", page_icon="🙂", layout="centered")


@st.cache_resource(show_spinner="Chargement de FaceNet...")
def charger_facenet():
    from keras_facenet import FaceNet
    return FaceNet()


@st.cache_resource(show_spinner="Entraînement du SVM...")
def entrainer_svm():
    data = np.load("embeddings.npz")
    X = normalize(data["facenet"])
    le = LabelEncoder().fit(data["labels"])
    svm = SVC(kernel="linear", probability=True, random_state=42).fit(X, le.transform(data["labels"]))
    return svm, le


@st.cache_resource
def charger_haar():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@st.cache_resource(show_spinner="Chargement de MTCNN...")
def charger_mtcnn():
    from mtcnn import MTCNN
    return MTCNN()


def mtcnn_disponible():
    try:
        import mtcnn  # noqa: F401
        return True
    except Exception:
        return False


def detecter_haar(img_rgb):
    gris = cv2.equalizeHist(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    boites = charger_haar().detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    boites = sorted(boites, key=lambda b: b[2] * b[3], reverse=True)
    return [tuple(int(v) for v in b) for b in boites]


def detecter_mtcnn(img_rgb):
    try:
        res = [r for r in charger_mtcnn().detect_faces(img_rgb) if r["confidence"] > 0.9]
    except ValueError:  # MTCNN plante quand il ne trouve aucun visage
        return detecter_haar(img_rgb)
    res.sort(key=lambda r: r["box"][2] * r["box"][3], reverse=True)
    return [tuple(max(v, 0) for v in r["box"]) for r in res]


def reduire(img, max_cote=1024):
    h, w = img.shape[:2]
    s = max_cote / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def recadrer(img, boite, taille=160, marge=0.1):
    x, y, w, h = boite
    mx, my = int(w * marge), int(h * marge)
    x1, y1 = max(x - mx, 0), max(y - my, 0)
    x2, y2 = min(x + w + mx, img.shape[1]), min(y + h + my, img.shape[0])
    visage = img[y1:y2, x1:x2]
    return None if visage.size == 0 else cv2.resize(visage, (taille, taille))


def reconnaitre(img_rgb, detecteur, seuil):
    embedder = charger_facenet()
    svm, le = entrainer_svm()
    img_rgb = reduire(img_rgb)
    sortie = img_rgb.copy()
    resultats = []
    for boite in detecteur(img_rgb):
        visage = recadrer(img_rgb, boite)
        if visage is None:
            continue
        emb = normalize(embedder.embeddings(np.array([visage])))
        probas = svm.predict_proba(emb)[0]
        i = int(np.argmax(probas))
        nom = le.classes_[i] if probas[i] >= seuil else "Inconnu"
        resultats.append((nom, float(probas[i]), dict(zip(le.classes_, probas))))
        x, y, w, h = boite
        ep = max(2, sortie.shape[1] // 300)
        cv2.rectangle(sortie, (x, y), (x + w, y + h), (0, 200, 0), ep)
        cv2.putText(sortie, f"{nom} {probas[i]:.2f}", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, sortie.shape[1] / 1000), (0, 200, 0), ep)
    return sortie, resultats


# ---------------------------------------------------------------- Interface
st.title("🙂 Reconnaissance faciale")
st.caption("FaceNet (embeddings) + SVM (classifieur). Projet pédagogique : utilisez uniquement des photos de personnes consentantes.")

try:
    _, le = entrainer_svm()
    st.sidebar.header("Personnes connues")
    for nom in le.classes_:
        st.sidebar.write(f"• {nom}")
except FileNotFoundError:
    st.error("Fichier `embeddings.npz` introuvable : placez-le à côté de `app.py`.")
    st.stop()

st.sidebar.header("Réglages")
options = ["Haar (rapide)"] + (["MTCNN (précis)"] if mtcnn_disponible() else [])
choix = st.sidebar.radio("Détecteur de visages", options)
detecteur = detecter_haar if choix.startswith("Haar") else detecter_mtcnn
seuil = st.sidebar.slider("Seuil de confiance (en dessous : « Inconnu »)", 0.0, 1.0, 0.6, 0.05)

onglet_photo, onglet_cam = st.tabs(["📁 Envoyer une photo", "📷 Webcam"])
fichier = onglet_photo.file_uploader("Choisissez une photo", type=["jpg", "jpeg", "png", "webp"])
capture = onglet_cam.camera_input("Prenez une photo")

source = fichier or capture
if source is not None:
    image = np.array(Image.open(source).convert("RGB"))
    sortie, resultats = reconnaitre(image, detecteur, seuil)
    st.image(sortie, use_container_width=True)
    if not resultats:
        st.warning("Aucun visage détecté. Essayez une photo plus nette et de face, ou changez de détecteur.")
    for nom, conf, probas in resultats:
        st.subheader(f"{nom}  ·  confiance {conf:.0%}")
        st.bar_chart(probas)
else:
    st.info("Envoyez une photo ou prenez-en une avec la webcam.")
