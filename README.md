# Color(not)Blind

Color(not)Blind is a web app I created for my end of boot camp project.

I wanted to develop a tool for visually impaired people, using image recognition and machine learning.

The app aim to determine the color and the type of clothing on a picture.

The easiest way for me at the moment, was to deploy the app using the Streamlit framework, but not yet on the Streamlit Cloud.


## How to use it

To run it locally on your computer, you need to:
- comply with the packages and their version (in requirement.txt)
- download the model (I used [Alexey Grigorev](https://github.com/alexeygrigorev)'s pre-trained model):

    `wget https://github.com/alexeygrigorev/mlbookcamp-code/releases/download/chapter7-model/xception_v4_large_08_0.894.h5`

- run the command: `streamlit run color\(not\)blind_app.py`
