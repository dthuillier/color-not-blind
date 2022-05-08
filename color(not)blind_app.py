import numpy as np
import streamlit as st
from PIL import Image
from gtts import gTTS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

from colorthief import ColorThief
from scipy.spatial import KDTree
from webcolors import CSS3_NAMES_TO_HEX
import webcolors
import io
import cv2


'''
# Color(not)Blind
#### _Identify your clothing item!_
'''


def main():
    # Sidebar Menu - Choose the function used to determine color
    st.sidebar.header('Color Function Menu')
    color_function = ['Hue', 'Hexadecimal']
    color_choice = st.sidebar.radio('Choose the color function :',
                                    color_function)

    # Idea to get input image from user
    #picture = st.camera_input("Take a picture")
    #if picture:
    #    st.image(picture)

    # File Uploader
    image_file = st.file_uploader("Please upload an image file",
                                  type=["jpg", "png", "jpeg"])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, use_column_width=True)

        # Save image file to be usable by model
        file_details = {"FileName": image_file.name,
                        "FileType": image_file.type}
        with open(image_file.name, "wb") as f:
            f.write(image_file.getbuffer())

        # Get the prediction
        prediction = function_predict_class(image_file.name)

        # Get the color of the image
        if color_choice == 'Hexadecimal':
            color = get_color(image_file.name)
        elif color_choice == 'Hue':
            color = get_hue(image_file.name)

        # Print prediction and color
        string = "This item is most likely a " + color + " " + prediction
        st.header(string)

        # Text to speech
        tts = gTTS(string)
        tts.save("descr.mp3")
        audio_file = open("descr.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/ogg", start_time=0)


@st.cache
def function_predict_class(image):
    '''
    Get image, pre-process the image, get clothing classification
    Argument :
        image (path)
    Returns :
        string category of clothing (label)
    '''

    # Labels dictionary
    labels = {
        0: 'dress',
        1: 'hat',
        2: 'longsleeve',
        3: 'outwear',
        4: 'pants',
        5: 'shirt',
        6: 'shoes',
        7: 'shorts',
        8: 'skirt',
        9: 't-shirt'
    }

    # Load pre-trained and stored model
    # Download the model : wget https://github.com/alexeygrigorev/mlbookcamp-code/releases/download/chapter7-model/xception_v4_large_08_0.894.h5
    model = keras.models.load_model('xception_v4_large_08_0.894.h5')

    # Pre-process the image
    image_size = (299, 299)
    img = load_img(image, target_size=(image_size))

    x = np.array(img)
    X = np.array([x])
    X = preprocess_input(X)

    # Get prediction
    pred = model.predict(X)
    label_pred = labels[pred[0].argmax()]

    return label_pred


def get_color(image):
    '''
    Get the Hexadecimal name of the dominant color of the image_file
    Argument :
        image (path)
    Returns :
        string name of the color
    '''

    # Load image
    img = load_img(image)
    img_vect = np.array(img)

    # Crop the image to get the center ROI
    height = int(img_vect.shape[0]/4)
    width = int(img_vect.shape[1]/4)

    roi = img_vect[height:3*height, width:3*width]

    # Save ROI as image file to be able to use it with ColorThief
    img_roi = Image.fromarray(roi)
    img_roi.save("img_roi.jpg")

    image_test = open("img_roi.jpg", 'rb')
    f = io.BytesIO(image_test.read())
    color_thief = ColorThief(f)
    dominant_color = color_thief.get_color(quality=1)

    return convert_rgb_to_names(dominant_color)


def convert_rgb_to_names(rgb_tuple):
    '''
    Convert RGB tuple to Hexadecimal name
    If no color associated, get the closest
    Argument :
        RGB color as a tuple
    Returns :
        String of Hexadecimal name
    '''

    # a dictionary of all the hex and their respective names in css3
    css3_db = webcolors.CSS3_NAMES_TO_HEX
    names = []
    rgb_values = []

    for color_name, color_hex in css3_db.items():
        names.append(color_name)
        rgb_values.append(webcolors.hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)

    return names[index]


def get_hue(image):
    '''
    Get the most frequent Hue of an image
    Argument :
        image (path)
    Returns :
        String name of the hue
    '''

    # Load image
    img = cv2.imread(image)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Crop the image to get the center ROI
    img_H = hsvImg[:, :, 0]
    height = int(img_H.shape[0]/4)
    width = int(img_H.shape[1]/4)

    roi = img_H[height:3*height, width:3*width]

    # Get the most frequent hue
    hue = (np.bincount(roi.flatten()).argmax())*2

    # Transform the value into a common name
    if 0 <= hue < 16:
        hue_label = 'red'
    elif 16 <= hue < 36:
        hue_label = 'orange'
    elif 36 <= hue < 71:
        hue_label = 'yellow'
    elif 71 <= hue < 80:
        hue_label = 'lime'
    elif 80 <= hue < 164:
        hue_label = 'green'
    elif 164 <= hue < 194:
        hue_label = 'cyan'
    elif 194 <= hue < 241:
        hue_label = 'blue'
    elif 241 <= hue < 261:
        hue_label = 'indigo'
    elif 261 <= hue < 271:
        hue_label = 'violet'
    elif 271 <= hue < 292:
        hue_label = 'purple'
    elif 292 <= hue < 328:
        hue_label = 'magenta'
    elif 328 <= hue < 345:
        hue_label = 'pink'
    elif 345 <= hue < 359:
        hue_label = 'red'

    return hue_label


if __name__ == '__main__':
    main()
