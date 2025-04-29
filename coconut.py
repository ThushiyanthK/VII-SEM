import streamlit as st
import os
import google.generativeai as genai
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gdown

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="🌴 Coconut Disease Diagnosis Bot", layout="centered")
st.title("🌴 Coconut Disease Diagnosis Chatbot 🤖")
st.write("Upload an image of a coconut tree or leaf and chat with our AI to diagnose diseases.")

# ------------------ GEMINI API KEY ------------------
genai.configure(api_key="AIzaSyA3VYu_hAB4T0QtUGbSJ2KTW7gIA1od1G8")  # Replace with your actual API Key

# ------------------ MODEL FILE SETUP ------------------
TREE_MODEL_PATH = "tree_model.keras"
LEAF_MODEL_PATH = "leaf_model.keras"

TREE_MODEL_ID = "1Qse74IbkhvuMCVytroGzvpT-9E6DuEU9"  # Replace with your tree model ID
LEAF_MODEL_ID = "1gUT8FKVCisPaFRl8efdavLmPzNx5fXLU"  # Replace with your leaf model ID

TREE_MODEL_URL = f"https://drive.google.com/uc?id={TREE_MODEL_ID}"
LEAF_MODEL_URL = f"https://drive.google.com/uc?id={LEAF_MODEL_ID}"

# ------------------ MODEL LOADING ------------------

def load_tree_model():
    if not os.path.exists(TREE_MODEL_PATH):
        st.info("Downloading tree model from Google Drive...")
        gdown.download(TREE_MODEL_URL, TREE_MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(TREE_MODEL_PATH, compile=False)

def load_leaf_model():
    if not os.path.exists(LEAF_MODEL_PATH):
        st.info("Downloading leaf model from Google Drive...")
        gdown.download(LEAF_MODEL_URL, LEAF_MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(LEAF_MODEL_PATH, compile=False)


tree_model = load_tree_model()
leaf_model = load_leaf_model()

# ------------------ DISEASE INFO ------------------
disease_info = {
    "BudRootDropping": {
        "cause": "Caused by fungal infection due to excess moisture.",
        "remedy": "Use fungicides and ensure proper drainage."
    },
    "BudRot": {
        "cause": "Caused by Phytophthora fungus affecting young palms.",
        "remedy": "Apply Bordeaux mixture and prune affected parts."
    },
    "LeafRot": {
        "cause": "Occurs due to fungal attack in humid conditions.",
        "remedy": "Use copper-based fungicides and remove infected leaves."
    },
    "StemBleeding": {
        "cause": "Caused by a fungal infection leading to dark gum exudation.",
        "remedy": "Scrape infected areas and apply fungicidal paste."
    }
}

leaf_disease_info = {
    "CCI_Caterpillars": {
        "cause": "Caused by caterpillar infestation feeding on the leaves.",
        "remedy": "Apply biological insecticides or neem-based sprays."
    },
    "CCI_Leaflets": {
        "cause": "Caused by nutritional deficiency or physical damage to leaflets.",
        "remedy": "Provide balanced nutrients and proper care."
    },
    "Healthy_Leaves": {
        "cause": "No disease detected. The leaves appear healthy.",
        "remedy": "Continue regular maintenance and monitoring."
    },
    "WCLWD_DryingofLeaflets": {
        "cause": "A symptom of root wilt disease leading to drying of leaflets.",
        "remedy": "Apply adequate fertilizers and organic matter to improve root health."
    },
    "WCLWD_Flaccidity": {
        "cause": "Caused by vascular disorder affecting water transport in the plant.",
        "remedy": "Improve irrigation practices and apply recommended nutrients."
    },
    "WCLWD_Yellowing": {
        "cause": "Initial stage of root wilt or nutritional deficiency.",
        "remedy": "Use magnesium and potassium-based fertilizers as prescribed."
    }
}

# ------------------ PREDICTION FUNCTION ------------------
def tree_predict_disease(image, model):
    img = image.resize((299, 299))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    if predicted_class >= len(disease_info):
        return "Unknown Disease", confidence

    return list(disease_info.keys())[predicted_class], confidence

def leaf_predict_disease(image, model):
    img = image.resize((299, 299))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    if predicted_class >= len(disease_info):
        return "Unknown Disease", confidence

    return list(leaf_disease_info.keys())[predicted_class], confidence

# ------------------ IMAGE UPLOAD SECTION ------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌴 Analyze Tree Image"):
            if "tree_model" not in st.session_state:
                st.session_state.tree_model = load_tree_model()
            label, confidence = tree_predict_disease(image, st.session_state.tree_model)
            response = f"✅ Predicted disease: {label}\n\n🎯 Confidence: {confidence:.2f}"
            st.session_state.last_predicted_disease = label
            st.session_state.last_disease_type = "tree"

    
            if label in disease_info:
                response += (
                    f"\n\n🧪 Cause: {disease_info[label]['cause']}"
                    f"\n💊 Remedy: {disease_info[label]['remedy']}"
                )
            else:
                response += "\n\n⚠ No additional information available for this disease."
            st.success(response)

    with col2:
        if st.button("🍃 Analyze Leaf Image"):
            if "leaf_model" not in st.session_state:
                st.session_state.leaf_model = load_leaf_model()
            label, confidence = leaf_predict_disease(image, st.session_state.leaf_model)
            print(label)
            st.session_state.last_predicted_disease = label
            st.session_state.last_disease_type = "leaf"
            response = f"✅ Predicted disease: {label}\n\n🎯 Confidence: {confidence:.2f}"
    
            if label in leaf_disease_info:
                response += (
                    f"\n\n🧪 Cause: {leaf_disease_info[label]['cause']}"
                    f"\n💊 Remedy: {leaf_disease_info[label]['remedy']}"
                )
            else:
                response += "\n\n⚠ No additional information available for this disease."
            st.success(response)

else:
    st.info("📸 Hello, farmer! Upload an image and select whether it's a tree or leaf for diagnosis.")
# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello, farmer! Upload an image and ask about coconut diseases!"
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ GEMINI AI CHATBOT ------------------
def ask_gemini(user_input):
    model = genai.GenerativeModel(
        "gemini-1.5-pro",
        system_instruction=(
            "You are a helpful assistant that only answers questions related to coconut diseases,their symptoms, causes, remedies, and coconut farming. If asked anything else, reply with:'I'm sorry, I can only help with coconut-related queries.'You must understand Tamil queries and respond in Tamil language only if the user expects a reply in tamil. If the user expects a reply in Tamil, give the reply in Tamil. Also, if the user input is in Tamil, understand it and reply in Tamil."
        )
    )

    history = [
        {"role": msg["role"], "parts": [msg["content"]]}
        for msg in st.session_state.get("messages", [])
    ]

    chat = model.start_chat(history=history)
    response = chat.send_message(user_input)
    return response.text

# ------------------ USER TEXT CHAT INPUT ------------------
if user_input := st.chat_input("Ask about coconut diseases or remedies..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    # If user refers to previous disease
    if ("this disease" in user_input.lower() or "above disease" in user_input.lower() or "predicted disease" in user_input.lower() or "மேற்கண்ட நோய்" in user_input.lower()) and "last_disease" in st.session_state:
        disease_name = st.session_state.last_predicted_disease
        disease_type = st.session_state.get("last_disease_type", "tree/leaf")
        print(disease_name)
        prompt = (
            f"The user previously analyzed a coconut {disease_type} image and it was predicted "
            f"to have the disease: '{disease_name}'.\n\n"
            f"The user now asked: \"{user_input}\"\n\n"
            f"Please provide a detailed explanation of this disease, including causes, symptoms, "
            f"remedies, and any preventive measures. If the user's question is in Tamil or implies Tamil, respond in Tamil. Or else reply in english"
        )
        response = ask_gemini(prompt)
    else:
        response = ask_gemini(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
