import io
import streamlit as st
import time
from io import BytesIO
from streamlit_image_comparison import image_comparison
from PIL import Image
from streamlit_js_eval import streamlit_js_eval
import requests
import base64
import os

from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000/colorize")
API_TOKEN = os.environ.get("API_TOKEN", "default") 

def colorize_from_api(img: Image) -> Image:
    """
    Sends an image to the FastAPI backend for colorization.
    """
    tic = time.time()
    
    if not img:
        st.error(selected_text["upload_error"])
        return

    with st.status(selected_text["colorizing"], expanded=True) as status:
        
        # Convert image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Prepare request
        headers = {"Token": API_TOKEN}
        json_data = {"image": img_str}
        
        st.write(selected_text["sending_to_api"])

        try:
            response = requests.post(FASTAPI_URL, headers=headers, json=json_data)
            
            if response.status_code == 200:
                data = response.json()
                # Decode the received base64 image
                colorized_img_str = data.get("image")
                colorized_img_bytes = base64.b64decode(colorized_img_str)
                colorized_image = Image.open(io.BytesIO(colorized_img_bytes))
                
                toc = time.time()
                status.update(label=selected_text["colorized_image"].format(toc - tic), state="complete", expanded=False)
                
                return colorized_image
            else:
                st.error(f"Error from API: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the colorization service: {e}")
            return None

def load_image(sample_image=None) -> Image:
    if sample_image:
        return Image.open(sample_image)

    uploaded_file = st.file_uploader(
        label=selected_text["upload_label"],
        label_visibility="collapsed"
    )
    try:
        if uploaded_file is not None:
            image_data = uploaded_file.getvalue()
            return Image.open(io.BytesIO(image_data))
        else:
            return None
    except Exception:
        st.error(selected_text["upload_error"])
        return None

def set_result_button_state():
    st.session_state['result'] = True

text = {
    "English 🇺🇸": {
        "title": "SEM in Colors 🔬⚡",
        "subtitle": "Colorize Scanning Electron Microscope images with AI & Computer Vision",
        "description": "At the moment, the model is capable of colorizing **pollen** images.\nThe goal is to progressively train it with more classes to become generalist.",
        "choose_sample": "Choose a sample image or upload your own",
        "upload_label": "Upload your image",
        "upload_error": 'Unable to process image. Try with a different file.',
        "sample_label": "Sample",
        "colorize_button": "Colorize",
        "colorizing": "Colorizing...",
        "sending_to_api": "Sending image to the colorization service...",
        "colorized_image": "Done in {:.2f} s ✅",
        "original": "Original",
        "colorized": "Colorized",
        "download_button": "Download",
        "filename": "Colorized"
    },
    "Português 🇧🇷": {
        "title": "MEV em Cores 🔬⚡",
        "subtitle": "Colorize imagens de Microscopia Eletrônica de Varredura com IA & Visão Computacional",
        "description": "No momento, o modelo é capaz de colorizar imagens de **pólens**.\nO objetivo é treiná-lo progressivamente com mais classes para torná-lo generalista.",
        "choose_sample": "Escolha uma imagem dentre as amostras ou faça upload da sua própria",
        "upload_label": "Fazer upload",
        "upload_error": 'Erro ao processar imagem. Tente novamente com outro arquivo.',
        "sample_label": "Amostra",
        "colorize_button": "Colorizar",
        "colorizing": "Colorizando...",
        "sending_to_api": "Enviando imagem para o serviço de colorização...",
        "colorized_image": "Concluído em {:.2f} s ✅",
        "original": "Original",
        "colorized": "Colorizada",
        "download_button": "Download",
        "filename": "Colorizada"
    }
}

if __name__ == "__main__":
    st.set_page_config(page_title=text['English 🇺🇸']["title"], layout="centered")

    languages = ["English 🇺🇸", "Português 🇧🇷"]
    language_selection = st.sidebar.selectbox("Select Language", languages)
    selected_text = text[language_selection]
    
    st.title(selected_text["title"])
    st.subheader(selected_text["subtitle"])
    st.markdown(selected_text["description"])

    sample_images = {
        f"{selected_text['sample_label']} 1": "./samples/Polen_1062_550X_rgb.png",
        f"{selected_text['sample_label']} 2": "./samples/PolenHibisco_300X-2_rgb.png",
        f"{selected_text['sample_label']} 3": "./samples/PolenHibisco_850X_rgb.png"
    }

    if "result" not in st.session_state:
        st.session_state["result"] = None

    screen_width = streamlit_js_eval(js_expressions='screen.width', key = 'SCR')
    if screen_width:
        comparison_width = 280 if screen_width < 400 else 700

    sample_selection = st.selectbox(selected_text["choose_sample"], [selected_text["upload_label"]] + list(sample_images.keys()))

    img = load_image(sample_image=sample_images.get(sample_selection))

    st.button(selected_text["colorize_button"], on_click=set_result_button_state)

    if st.session_state["result"] and img is not None:
        colorized = colorize_from_api(img)
        
        if colorized:
            image_comparison(
                img.convert('L'),
                colorized,
                selected_text["original"],
                selected_text["colorized"],
                make_responsive=False,
                width=comparison_width
            )

            buffer = BytesIO()
            colorized.save(buffer, format="PNG")
            byte_im = buffer.getvalue()
            st.download_button(label=selected_text["download_button"], data=byte_im, file_name=f"{selected_text['filename']}.png", mime="image/png")