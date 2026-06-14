import io
import streamlit as st
import time
import requests
import base64
import os
import hashlib
import numpy as np
from io import BytesIO
from streamlit_image_comparison import image_comparison
from PIL import Image
from streamlit_js_eval import streamlit_js_eval
from skimage.color import rgb2hsv, hsv2rgb

from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000/colorize")
API_TOKEN = os.environ.get("API_TOKEN", "default") 
MODEL_COLOR_HEX = "#e7b21f"


def hex_to_hue(color: str) -> float:
    color = color.lstrip("#")
    rgb = np.array([[[int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)]]], dtype=np.float32)
    return float(rgb2hsv(rgb)[0, 0, 0])


def apply_color_adjustments(
    colorized_img: Image.Image,
    target_color: str,
    saturation_percent: int,
) -> Image.Image:
    source_hue = hex_to_hue(MODEL_COLOR_HEX)
    target_hue = hex_to_hue(target_color)
    hue_shift = target_hue - source_hue

    rgb = np.asarray(colorized_img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = rgb2hsv(rgb)

    colored_pixels = hsv[:, :, 1] > 0.05
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    hue[colored_pixels] = (hue[colored_pixels] + hue_shift) % 1.0
    saturation[colored_pixels] = np.clip(saturation[colored_pixels] * (saturation_percent / 100), 0.0, 1.0)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = saturation

    adjusted_rgb = hsv2rgb(hsv)
    return Image.fromarray((np.clip(adjusted_rgb, 0.0, 1.0) * 255).astype(np.uint8))


def image_cache_key(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def normalize_uploaded_image(uploaded_file) -> Image.Image:
    image_data = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(image_data))
    extension = os.path.splitext(uploaded_file.name)[1].lower()

    if extension in (".tif", ".tiff"):
        if getattr(img, "is_animated", False):
            img.seek(0)

        buffer = BytesIO()
        png_img = img.copy()
        if png_img.mode == "CMYK":
            png_img = png_img.convert("RGB")
        png_img.save(buffer, format="PNG")
        buffer.seek(0)
        img = Image.open(buffer)

    return img.copy()

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
        label_visibility="collapsed",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
    try:
        if uploaded_file is not None:
            return normalize_uploaded_image(uploaded_file)
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
        "description": "",
        "choose_sample": "Choose a sample image or upload your own",
        "upload_label": "Upload your image",
        "upload_error": 'Unable to process image. Try with a different file.',
        "target_color": "Target color",
        "saturation": "Saturation",
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
        "description": "",
        "choose_sample": "Escolha uma imagem dentre as amostras ou faça upload da sua própria",
        "upload_label": "Fazer upload",
        "upload_error": 'Erro ao processar imagem. Tente novamente com outro arquivo.',
        "target_color": "Cor alvo",
        "saturation": "Saturação",
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
    st.set_page_config(page_title=text['Português 🇧🇷']["title"], layout="centered")

    languages = ["Português 🇧🇷", "English 🇺🇸"]
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
    if "colorized_base" not in st.session_state:
        st.session_state["colorized_base"] = None
    if "colorized_key" not in st.session_state:
        st.session_state["colorized_key"] = None
    if "requested_key" not in st.session_state:
        st.session_state["requested_key"] = None

    comparison_width = 700
    screen_width = streamlit_js_eval(js_expressions='screen.width', key = 'SCR')
    if screen_width:
        comparison_width = 280 if screen_width < 400 else 700

    sample_selection = st.selectbox(selected_text["choose_sample"], [selected_text["upload_label"]] + list(sample_images.keys()))

    img = load_image(sample_image=sample_images.get(sample_selection))
    current_image_key = image_cache_key(img) if img is not None else None

    colorize_clicked = st.button(selected_text["colorize_button"])
    if colorize_clicked:
        if img is None:
            st.error(selected_text["upload_error"])
        else:
            set_result_button_state()
            st.session_state["requested_key"] = current_image_key

    if st.session_state["result"] and img is not None and st.session_state["requested_key"] == current_image_key:
        if st.session_state["colorized_key"] != current_image_key:
            colorized_base = colorize_from_api(img)
            if colorized_base:
                st.session_state["colorized_base"] = colorized_base
                st.session_state["colorized_key"] = current_image_key
            else:
                st.session_state["colorized_base"] = None

        colorized = st.session_state["colorized_base"]
        
        if colorized:
            color_col, saturation_col = st.columns(2)
            with color_col:
                target_color = st.color_picker(selected_text["target_color"], MODEL_COLOR_HEX)
            with saturation_col:
                saturation_percent = st.slider(selected_text["saturation"], 0, 200, 100, 1, "%d%%")

            colorized = apply_color_adjustments(colorized, target_color, saturation_percent)

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