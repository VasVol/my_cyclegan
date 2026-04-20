import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as tr
from PIL import Image

from model import CycleGAN


st.set_page_config(
    page_title="CycleGAN Translators",
    page_icon="🌀",
    layout="wide",
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TASKS = {
    "horse2zebra": {
        "title": "Horse ↔ Zebra",
        "description": "Translate horse photos into zebras and zebra photos into horses.",
        "button_label": "Open Horse ↔ Zebra",
        "checkpoint_path": "./horses_best.pt",
        "model_params": {
            "img_channels_a": 3,
            "img_channels_b": 3,
            "ngf": 64,
            "ndf": 64,
            "image_size": 256,
            "num_res_blocks": 9,
            "init_gain": 0.02,
        },
        "domain_a_name": "Horse",
        "domain_b_name": "Zebra",
        "preview_image_a": "./horse_preview.png",
        "preview_image_b": "./zebra_preview.png",
    },
    "face2cartoon": {
        "title": "Sketch Face ↔ Cartoon Face",
        "description": "Translate pencil face sketches into colored cartoon faces and back.",
        "button_label": "Open Sketch ↔ Cartoon",
        "checkpoint_path": "./faces_best.pt",
        "model_params": {
            "img_channels_a": 3,
            "img_channels_b": 3,
            "ngf": 64,
            "ndf": 64,
            "image_size": 256,
            "num_res_blocks": 9,
            "init_gain": 0.02,
        },
        "domain_a_name": "Sketch Face",
        "domain_b_name": "Cartoon Face",
        "preview_image_a": "./sketch_face_preview.png",
        "preview_image_b": "./cartoon_face_preview.png",
    },
}


def build_transform(image_size: int):
    return tr.Compose([
        tr.Resize((image_size, image_size)),
        tr.ToTensor(),
        tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


@st.cache_resource(show_spinner=False)
def load_model(task_key: str):
    task = TASKS[task_key]
    checkpoint_path = Path(task["checkpoint_path"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Update checkpoint_path in TASKS inside the Streamlit file."
        )

    model = CycleGAN(**task["model_params"]).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_tensor = image_tensor.detach().cpu().clone()
    image_tensor = image_tensor * 0.5 + 0.5
    image_tensor = image_tensor.clamp(0, 1)
    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)


def run_translation(task_key: str, uploaded_file, direction: str):
    task = TASKS[task_key]
    model = load_model(task_key)

    image = Image.open(uploaded_file).convert("RGB")
    transform = build_transform(task["model_params"]["image_size"])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if direction == "A2B":
            output_tensor = model.G_AB(input_tensor)[0]
        else:
            output_tensor = model.G_BA(input_tensor)[0]

    output_image = tensor_to_pil(output_tensor)
    return image, output_image


def go_home():
    st.session_state["selected_task"] = None


def select_task(task_key: str):
    st.session_state["selected_task"] = task_key


def render_preview_pair(task: dict):
    left, right = st.columns(2)

    preview_a = Path(task["preview_image_a"])
    preview_b = Path(task["preview_image_b"])

    with left:
        st.caption(task["domain_a_name"])
        if preview_a.exists():
            st.image(str(preview_a), use_container_width=True)
        else:
            st.info(f"Add preview image here:\n\n`{preview_a}`")

    with right:
        st.caption(task["domain_b_name"])
        if preview_b.exists():
            st.image(str(preview_b), use_container_width=True)
        else:
            st.info(f"Add preview image here:\n\n`{preview_b}`")


def render_home():
    st.title("CycleGAN Translators")
    st.write("Choose which translator you want to use.")

    col1, col2 = st.columns(2, gap="large")
    task_keys = list(TASKS.keys())

    for col, task_key in zip([col1, col2], task_keys):
        task = TASKS[task_key]
        with col:
            st.subheader(task["title"])
            render_preview_pair(task)
            st.write(task["description"])
            st.button(
                task["button_label"],
                key=f"open_{task_key}",
                use_container_width=True,
                on_click=select_task,
                args=(task_key,),
            )


def render_task(task_key: str):
    task = TASKS[task_key]
    st.title(task["title"])

    top_left, top_right = st.columns([1, 5])
    with top_left:
        st.button("← Back", on_click=go_home, use_container_width=True)
    with top_right:
        st.write(task["description"])

    checkpoint_path = Path(task["checkpoint_path"])
    if not checkpoint_path.exists():
        st.error(
            "Checkpoint file was not found.\n\n"
            f"Expected path: `{checkpoint_path}`"
        )
        return

    st.success(f"Model device: {DEVICE}")

    direction = st.radio(
        "Translation direction",
        options=["A2B", "B2A"],
        format_func=lambda x: (
            f"{task['domain_a_name']} → {task['domain_b_name']}"
            if x == "A2B"
            else f"{task['domain_b_name']} → {task['domain_a_name']}"
        ),
        horizontal=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload an image to run the translator.")
        return

    if st.button("Translate", type="primary", use_container_width=True):
        with st.spinner("Running model..."):
            input_image, output_image = run_translation(task_key, uploaded_file, direction)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(input_image, use_container_width=True)
        with col2:
            st.subheader("Output")
            st.image(output_image, use_container_width=True)

        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        st.download_button(
            "Download output",
            data=buf.getvalue(),
            file_name=f"{task_key}_{direction.lower()}_output.png",
            mime="image/png",
            use_container_width=True,
        )


if "selected_task" not in st.session_state:
    st.session_state["selected_task"] = None

selected_task = st.session_state["selected_task"]

if selected_task is None:
    render_home()
else:
    render_task(selected_task)
