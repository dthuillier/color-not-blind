# Color(not)Blind 👓

Color(not)Blind is a web app I created for my end of boot camp project.

I wanted to develop a tool for visually impaired people, using image recognition and machine learning.

The app aims to determine the color and the type of clothing on a picture. 👕 👖 👗 🧥 🎨

The easiest way for me at the moment, was to deploy the app using the Streamlit framework, but not yet on the Streamlit Cloud.


## How to use it ⚙️

To run it locally on your computer, you need to execute those commands:
- Clone the repository: `git clone https://github.com/dthuillier/color-not-blind.git`
- Install the necessary packages from your working repository: `pip install -r requirements.txt`
- Download the model in your working repository (I used [Alexey Grigorev](https://github.com/alexeygrigorev)'s pre-trained model):

    `wget https://github.com/alexeygrigorev/mlbookcamp-code/releases/download/chapter7-model/xception_v4_large_08_0.894.h5`

- And run the app: `streamlit run color\(not\)blind_app.py`

**Questions?** 🔎

Thank you for checking my work out! ☀️

Don't hesitate to contact me if you have questions and have a nice day! 📫

---
`wget` is already installed on Ubuntu. For other operating systems: download Wget [here](https://www.gnu.org/software/wget/).  
