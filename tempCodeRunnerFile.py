import os
import json
import librosa
import cv2
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')


def streamlit_config():

    st.set_page_config(page_title='Classification', layout='centered')

    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    st.markdown(f'<h1 style="text-align: center;">Bird Sound Classification</h1>',
                unsafe_allow_html=True)
    add_vertical_space(4)


streamlit_config()

AUDIO_DIR = 'D:\\Projects\\Identifying Bird speciy from Audio\\Bird-Sound-Classification-using-Deep-Learning\\Sample_Audio\\norcar'        # Change this to your audio folder path
OUTPUT_DIR = 'spectrogram'            

def create_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.axis('off')
    librosa.display.specshow(S_DB, sr=sr, cmap='magma')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def convert_audio_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in tqdm(os.listdir(AUDIO_DIR), desc="Converting audio files"):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            label = filename.split('_')[0]  # Extract species from filename
            input_path = os.path.join(AUDIO_DIR, filename)
            output_folder = os.path.join(OUTPUT_DIR, label)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, filename.replace('.wav', '.png').replace('.mp3', '.png'))
            create_spectrogram(input_path, output_path)

if __name__ == '__main__':
    convert_audio_files()

def prediction(audio_file):
    import librosa.display
    import matplotlib.pyplot as plt
    from PIL import Image
    import uuid

    # Generate a unique temp filename to avoid conflicts
    temp_img_path = f"temp_spec_{uuid.uuid4().hex}.png"

    # Convert uploaded audio to mel spectrogram and save as image
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.axis('off')
    librosa.display.specshow(S_DB, sr=sr, cmap='magma')
    plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Load the spectrogram image and preprocess
    img = Image.open(temp_img_path).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

    # Load the CNN model trained on spectrograms
    model = tf.keras.models.load_model('model.h5')
    pred = model.predict(img_array)[0][0]  # Adjust depending on output shape

    confidence = round(pred * 100, 2)
    st.markdown(f'<h4 style="text-align: center; color: orange;">{confidence}% Match Found</h4>', 
                unsafe_allow_html=True)

    if pred >= 0.8:
        st.markdown(f'<h3 style="text-align: center; color: green;">This audio likely belongs to the known species.</h3>', 
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<h3 style="text-align: center; color: red;">Species not recognized confidently.</h3>', 
                    unsafe_allow_html=True)

    # Clean up temporary file
    os.remove(temp_img_path)
    


_,col2,_  = st.columns([0.1,0.9,0.1])
with col2:
    input_audio = st.file_uploader(label='Upload the Audio', type=['mp3', 'wav'])

if input_audio is not None:

    _,col2,_ = st.columns([0.2,0.8,0.2])
    with col2:
        prediction(input_audio)
