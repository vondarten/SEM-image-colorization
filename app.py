import gc
import io
import streamlit as st
import numpy as np
import torch
import time
import random
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_image_comparison import image_comparison
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torchvision import models
from fastai.torch_core import TensorBase
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from typing import Dict
from streamlit_js_eval import streamlit_js_eval

class UNetColorizer(DynamicUnet):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: TensorBase) -> TensorBase:
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            st.session_state["feature_maps"].append(nres)
            res.orig, nres.orig = None, None
            res = nres
        return res

def lab_to_rgb(L, ab) -> np.ndarray:

    """
    Convert images from LAB color space to RGB color space.

    This function denormalizes the LAB images and converts them to RGB format.

    Args:
        L (torch.Tensor): A tensor of shape (N, 1, H, W) where N is the number of images,
                          H is the height, and W is the width. The tensor contains 
                          the L (luminance) channel of LAB images, normalized to the range [-1, 1].
        ab (torch.Tensor): A tensor of shape (N, 2, H, W) where N is the number of images,
                           H is the height, and W is the width. The tensor contains 
                           the a and b channels of LAB images, normalized to the range [-1, 1].

    Returns:
        np.ndarray: A NumPy array of shape (N, H, W, 3) containing the RGB images, 
                    where N is the number of images, H is the height, W is the width, 
                    and 3 represents the RGB channels.
    """

    # Denormalize
    L = (L + 1.0) * 50.0
    ab = ab * 255.0 - 128.0

    Lab = torch.cat([L, ab], dim=1).cpu().numpy()
    rgb_imgs = []

    for img in Lab:
        img = np.transpose(img, (1, 2, 0)) 
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def rgb_to_lab(img: np.ndarray) -> Dict:

    """
    Convert an RGB image to LAB color space.

    This function takes an RGB image, converts it to LAB color space, and normalizes
    the LAB channels to a specific range.

    Args:
        img (np.ndarray): A NumPy array representing the RGB image with shape 
                          (height, width, 3). The image can be in RGBA format, 
                          but it will be converted to RGB if necessary.

    Returns:
        Dict: A dictionary with two keys:
            - 'L': A tensor of shape (1, 1, height, width) representing the luminance 
                    channel of the LAB image, normalized to the range [-1, 1].
            - 'ab': A tensor of shape (2, 1, height, width) representing the a and b 
                    channels of the LAB image, normalized to the range [-1, 1].
    """

    img = img.convert('RGB')

    img_lab = rgb2lab(np.array(img))

    # Reshape to (image_size, image_size, channels)
    img_lab = torch.from_numpy(img_lab).permute(2, 0, 1).float()
    img_lab = torch.unsqueeze(img_lab, 1)

    L = img_lab[0]
    ab = img_lab[1:]

    # Normalization: -1.0 <= x <= 1.0
    L = (L / 50.0) - 1.0
    ab = (ab + 128.0) / 255.0
    
    return {'L': L, 'ab': ab}

def colorize(img: Image) -> Image:

    """
    Colorize a grayscale image using a trained UNet colorization model.

    This function takes a grayscale image and uses a colorization model to add color
    to it. It involves several steps: converting the image to LAB color space, resizing
    it for model input, running the model to generate color channels, upsampling the
    generated color channels, and combining them with the original luminance channel.

    Args:
        model (UNetColorizer): The UNet model used for colorizing the image.
        img (Image): A PIL Image object representing the grayscale image to be colorized.

    Returns:
        Image: A PIL Image object representing the colorized image.
    """

    tic = time.time()

    if not img:
        st.error(selected_text["upload_error"])
        return

    with st.status(selected_text["loading_model"], expanded=True) as status:
        model = load_model()
        status.update(label=selected_text["loaded_model"].format(sum(p.numel() for p in model.parameters())), state="complete", expanded=False)

    with st.status(selected_text["colorizing"], expanded=True) as status:

        st.session_state["feature_maps"] = []
        
        orig_img = img.convert('L')
        width, height = orig_img.size
        
        st.write(selected_text["preparing"])
        ### Convert the original image to LAB and keep the original L channel
        orig_lab = rgb_to_lab(img)
        orig_L = orig_lab['L']

        ### Resize the original image and convert to LAB for model input
        resized_img = transforms.Resize((384, 384), Image.BICUBIC)(img)
        resized_lab = rgb_to_lab(resized_img)
        
        sample = {'L': resized_lab['L'].unsqueeze(0).to('cpu'), 
                'ab': resized_lab['ab'].permute(1, 0, 2, 3).to('cpu')
                }

        model.eval()

        st.write(selected_text["forward_prop"])

        with torch.inference_mode():
            model_ab = model(sample['L'])
        
        st.write(selected_text["interpolating"])
        ### Upsample the generated ab channels to the original image size
        model_ab = torch.nn.functional.interpolate(model_ab, size=(height, width), mode='bilinear')

        st.write(selected_text["converting"])
        ### Combine the original L channel with the upsampled ab channels
        colorized_img = lab_to_rgb(orig_L.unsqueeze(0), model_ab).squeeze(0)

        toc = time.time()
        
        status.update(label=selected_text["colorized_image"].format(toc-tic), state="complete", expanded=False)

        gc.collect()

        return Image.fromarray((colorized_img * 255).astype(np.uint8))


def load_image(sample_image=None) -> Image:

    """
    Load an image from a file or user upload.

    This function attempts to load an image from a specified file path or from a file
    uploaded by the user. If a sample image path is provided, it will load the image
    from that path. Otherwise, it will use a file uploader widget to allow the user to
    upload an image.

    Args:
        sample_image (str, optional): The file path of the sample image to be loaded. 
                                      If not provided, the function will use a file 
                                      uploader for user uploads.

    Returns:
        Image: A PIL Image object representing the loaded image. If no image is uploaded 
               or an error occurs, returns None.
    """

    if sample_image:
        return Image.open(sample_image)

    uploaded_file = st.file_uploader(label='')
    try:
        if uploaded_file is not None:
            image_data = uploaded_file.getvalue()
            return Image.open(io.BytesIO(image_data))
        else:
            return None
    except Exception:
        st.error(selected_text["upload_error"])
        return None

@st.cache_resource
def load_model() -> UNetColorizer:
    """
    Load a pre-trained UNet colorization model.

    This function initializes a UNetColorizer model with a ResNet-18 backbone and loads
    pre-trained weights from a specified checkpoint. The model is set up for colorizing
    images with dimensions (384, 384).

    Returns:
        UNetColorizer: The loaded UNetColorizer model ready for inference.
    """

    basemodel = models.resnet18()
    body = create_body(basemodel, n_in=1, pretrained=False, cut=-2) 
    model = UNetColorizer(body, 2, (384, 384), self_attention=False, act_cls=torch.nn.Mish).to('cpu')
    experiment = 'gan-unet-resnet18-just-mish-4-2024-07-18-17:45:12'
    saved_model = torch.load(f"./experiments/{experiment}/model.pth", map_location='cpu')
    model.load_state_dict(saved_model['model_state_dict'])

    return model

def plot_feature_maps() -> None:
    """
    Plot and display feature maps from the current session state.

    This function selects a random depth from the stored feature maps and visualizes
    the feature maps at that depth. If the depth contains 9 or more feature maps, 
    it randomly selects 9 and plots them in a 3x3 grid. If there are exactly 2 
    feature maps at the chosen depth, they are plotted side by side.

    The plots are displayed using Streamlit's `st.pyplot` function.

    Returns:
        None
    """

    depth_to_choose = random.randint(0, len(st.session_state["feature_maps"]) - 1)

    depth_shape = st.session_state["feature_maps"][depth_to_choose].shape

    if depth_shape[1] >= 9:

        ### Extract the feature maps at the chosen depth
        feature_maps = st.session_state["feature_maps"][depth_to_choose].detach().squeeze(0).numpy()

        ### Select 9 random feature maps
        random_indices = random.sample(range(depth_shape[1]), 9)
        selected_feature_maps = feature_maps[random_indices, :, :]

        ### Plot the 9 feature maps in a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))

        for i, (ax, idx) in enumerate(zip(axes.flat, random_indices)):
            im = ax.imshow(selected_feature_maps[i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'#{idx} - {selected_text["tensor_shape"]}: {selected_feature_maps[i].shape}', fontsize=24)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'9/{depth_shape[1]} {selected_text["feature_maps_plot_title"]} {depth_to_choose}', fontsize=32)
        plt.subplots_adjust(top=0.90)
        st.pyplot(fig)

    elif depth_shape[1] == 2:

        ### Extract the feature maps at the chosen depth
        feature_maps = st.session_state["feature_maps"][depth_to_choose].detach().squeeze(0).numpy()

        ### Plot the 2 feature maps side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))

        for i, (ax, idx) in enumerate(zip(axes, range(2))):
            im = ax.imshow(feature_maps[i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'#{idx} - {selected_text["tensor_shape"]}: {feature_maps[i].shape}', fontsize=24)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'{selected_text["feature_maps_plot_title"]} {depth_to_choose}', fontsize=32)
        plt.subplots_adjust(top=0.90)
        st.pyplot(fig)

        ### Release memory
        for feature_map in st.session_state["feature_maps"]:
            del feature_map

    gc.collect()


def set_result_button_state():
    st.session_state['result'] = True
    st.session_state['show_fmaps'] = False

def set_show_fmaps_button_state():
    st.session_state['show_fmaps'] = True

# Define text for both languages
text = {
    "English 🇺🇸": {
        "title": "SEM in Colors 🔬⚡",
        "subtitle": "Colorize Scanning Electron Microscope images with AI & Computer Vision",
        "description": "At the moment, the model is capable of colorizing **pollen** images.\nThe goal is to progressively train it with more classes to become generalist.",
        "loading_model": "Loading model...",
        "loaded_model": "Loaded Model - {:,} parameters! ✅",
        "choose_sample": "Choose a sample image or upload your own",
        "upload_label": "Upload your image",
        "upload_error": 'Unable to process image. Try with a different file.',
        "sample_label": "Sample",
        "colorize_button": "Colorize",
        "colorizing": "Colorizing...",
        "preparing": "Preparing input image for the model...",
        "forward_prop": "Model inference...",
        "interpolating": "Interpolating generated AB channels back to original size...",
        "converting": "Converting from LAB back to RGB...",
        "colorized_image": "Done in {:.2f} s ✅",
        "original": "Original",
        "colorized": "Colorized",
        "see_feature_maps": "See feature maps",
        "feature_maps_plot_title": "Feature Maps at Depth",
        "getting_feature_maps": 'Getting feature maps',
        "tensor_shape": "Shape",
        "download_button": "Download",
        "filename": "Colorized"
    },
    "Português 🇧🇷": {
        "title": "MEV em Cores 🔬⚡",
        "subtitle": "Colorize imagens de Microscopia Eletrônica de Varredura com IA & Visão Computacional",
        "description": "No momento, o modelo é capaz de colorizar imagens de **pólens**.\nO objetivo é treiná-lo progressivamente com mais classes para torná-lo generalista.",
        "loading_model": "Carregando o modelo...",
        "loaded_model": "Modelo Carregado - {:,} parâmetros! ✅",
        "choose_sample": "Escolha uma imagem dentre as amostras ou faça upload da sua própria",
        "upload_label": "Fazer upload",
        "upload_error": 'Erro ao processar imagem. Tente novamente com outro arquivo.',
        "sample_label": "Amostra",
        "colorize_button": "Colorizar",
        "colorizing": "Colorizando...",
        "preparing": "Preparando a imagem para o modelo...",
        "forward_prop": "Inferência...",
        "interpolating": "Interpolando os canais AB gerados de volta às dimensões originais...",
        "converting": "Convertendo de LAB para RGB...",
        "colorized_image": "Concluído em {:.2f} s ✅",
        "original": "Original",
        "colorized": "Colorizada",
        "see_feature_maps": "Ver mapas de características",
        "feature_maps_plot_title": "Mapas de Características na Profundidade",
        "getting_feature_maps": 'Gerando Mapas de Características',
        "tensor_shape": "Dimensões",
        "download_button": "Download",
        "filename": "Colorizada"
    }
}

if __name__ == "__main__":

    st.set_page_config(page_title=text['English 🇺🇸']["title"], layout="centered")

    ### Set language options
    languages = ["English 🇺🇸", "Português 🇧🇷"]
    language_selection = st.sidebar.selectbox("Select Language", languages)

    ### Get the selected language text
    selected_text = text[language_selection]
    st.title(selected_text["title"])
    st.subheader(selected_text["subtitle"])
    st.markdown(selected_text["description"])

    ### Sample images
    sample_images = {
        f"{selected_text['sample_label']} 1": "./samples/Polen_1062_550X_rgb.png",
        f"{selected_text['sample_label']} 2": "./samples/PolenHibisco_300X-2_rgb.png",
        f"{selected_text['sample_label']} 3": "./samples/PolenHibisco_850X_rgb.png"
    }

    if "feature_maps" not in st.session_state:
        st.session_state["feature_maps"] = []

    if "result" not in st.session_state:
        st.session_state["result"] = None

    if "show_fmaps" not in st.session_state:
        st.session_state["show_fmaps"] = None

    screen_width = streamlit_js_eval(js_expressions='screen.width', key = 'SCR')

    if screen_width:
        comparison_width = 280 if screen_width < 400 else 700

    sample_selection = st.selectbox(selected_text["choose_sample"], [selected_text["upload_label"]] + list(sample_images.keys()))

    if sample_selection == selected_text["upload_label"]:
        img = load_image()
    else:
        img = load_image(sample_image=sample_images[sample_selection])

    result = st.button(selected_text["colorize_button"], on_click=set_result_button_state)

    ### Perform colorization if the button was pressed
    if st.session_state["result"] and img is not None:
        tic = time.time()
        colorized = colorize(img)
        toc = time.time()

        # Show the image comparison
        image_comparison(
            img.convert('L'),
            colorized,
            selected_text["original"],
            selected_text["colorized"],
            make_responsive=False,
            width=comparison_width
        )

        ### Display the "Show feature maps" button and handle its state
        show_fmaps = st.button(selected_text["see_feature_maps"], on_click=set_show_fmaps_button_state)
        
        ### Show feature maps if the button was pressed
        if st.session_state["show_fmaps"]:
            with st.spinner(selected_text["getting_feature_maps"]):
                time.sleep(0.3)
                plot_feature_maps()

        ### Provide a download button for the colorized image
        buffer = BytesIO()
        colorized.save(buffer, format="PNG")
        byte_im = buffer.getvalue()
        st.download_button(label=selected_text["download_button"], data=byte_im, file_name=f"{selected_text['filename']}.png", mime="image/png")