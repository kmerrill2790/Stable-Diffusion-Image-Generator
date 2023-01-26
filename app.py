import streamlit as st
import banana_dev as banana
import replicate
import base64
from io import BytesIO
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import urllib.request
import random
from zipfile import ZipFile
import shutil
import os
import pathlib

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True
if check_password():
    def merge_inputs(callInputs: dict, modelInputs: dict):
        return(modelInputs.update(callInputs))

    def decode_and_save1(image_byte_string: str, modelInputs: dict):
        seed = str(modelInputs['seed'])
        image_encoded = image_byte_string.encode("utf-8")
        image_bytes = BytesIO(base64.b64decode(image_encoded))
        image = PIL.Image.open(image_bytes)
        metadata = PngInfo()
        dict2text = json.dumps(modelInputs, indent = 4)
        metadata.add_text('parameters',dict2text)
        canvas.image(image)
        image.save(f'{seed}.png', pnginfo=metadata)
        status.success(f'Successfully rendered a {image.width}x{image.height} {image.format} image with seed:\n{seed}', icon="✅")
        with open(f'{seed}.png', 'rb') as f:
            download_col.download_button('Download Image', f, file_name=f'{seed}.png', key='single')

    def decode_and_save_multi(image_byte_strings: str, modelInputs: dict):
        seed = str(modelInputs['seed'])
        num_images = modelInputs['num_images_per_prompt']
        filenames = ["model_inputs.json"]
        for idx, image_byte_string in enumerate(image_byte_strings):
            image_encoded = image_byte_string.encode("utf-8")
            image_bytes = BytesIO(base64.b64decode(image_encoded))
            image = PIL.Image.open(image_bytes)
            metadata = PngInfo()
            dict2text = json.dumps(modelInputs, indent = 4)
            metadata.add_text('parameters',dict2text)
            image.save(f'{seed}_{idx}.png', pnginfo=metadata)
            filenames.append(f'{seed}_{idx}.png')
        with open("model_inputs.json", "w") as outfile:
            json.dump(modelInputs, outfile, indent = 6)
        with ZipFile(f'{seed}.zip', mode="w") as archive:
            for filename in filenames:
                archive.write(filename)
        with open(f'{seed}.zip', 'rb') as f:
            download_col.download_button('Download Images', f, file_name=f'{seed}.zip', key='multi') 
        status.success(f'Successfully rendered ({num_images}) {image.width}x{image.height} {image.format} images with seed:\n{seed}', icon="✅")
        del filenames[0]
        canvas.image(filenames, use_column_width='auto')

    def b64encode_file(filename: str):
        path = pathlib.Path(filename)
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("ascii")

    def img_lab_download(image_url: str, image_filename: str):
        urllib.request.urlretrieve(image_url, image_filename)
        image = PIL.Image.open(image_filename)
        image.save(image_filename)
        with open(image_filename, "rb") as file:
            download_image = download_col.download_button("Download Image", data=file, file_name=image_filename, mime="image/png")
        status.success('Post Processing Complete', icon="✅")
        canvas.image(image)

    st.set_page_config(page_title="Stable Diffusion", page_icon=":frame_with_picture:")
    hide_menu_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.title("Stable Diffusion Image Generator 🎨 ")
    api_key = st.secrets["banana_key"]
    model_keys = {
        'Stable 1.5': ('runwayml/stable-diffusion-v1-5','956d8b5d-f94f-4fdb-8080-84747e3d69b4'), 
        'Stable 2.1': ('stabilityai/stable-diffusion-2-1','bcc9241e-a891-430f-a0d2-dc584cc7da74'), 
        'OpenJourney V2': ('prompthero/openjourney-v2','cf5cab8b-00d7-4c52-bd12-53fa45c478ad'),
        'WikiArt V2':('ckpt/sd-wikiart-v2','392c6e8e-74f9-444f-9ef1-138fbd6cbcc7'), 
        'Stable Analog': ('wavymulder/Analog-Diffusion','d1c03b36-92ec-4868-bb21-311d66f03499'),
        'portrait+ style':('wavymulder/portraitplus','0006f0e9-3b30-4cde-85a8-8396b594b900'), 
        'Dreamlike Photoreal 2.0': ('dreamlike-art/dreamlike-photoreal-2.0', 'a42a5465-c4ce-40a4-9048-04a4bb6d19a9'),
        'Photorealistic Fuen V1': ('claudfuen/photorealistic-fuen-v1', '792137f3-feab-4aef-93a4-50f9cd5b7269'),
        'classicnegative photo':('BudFactory/classicnegative','d08fb007-9081-4f77-bff6-2c839c850ca6'),
        'Protogen V2.2':('ckpt/Protogen_V2.2','88b9bfc4-902c-46d1-a96f-2af28122cda3'),
        'Protogen V3.4':('Protogen_x3.4','b60a589f-b8db-4ef9-87e1-f61bc9a19adb'),
        'Protogen V5.3':('Protogen_x5.3','20a06acd-908f-4619-984e-f62b22d04241'),
        'Protogen V5.8':('Protogen_x5.8','aa7646ab-a5b7-47f2-9600-23678d433496'),
        'f222': ('lilpotat/f2','575fdd56-047d-4aea-a9b6-ab1e2996320c'),
        'Realistic Vision V1.1':('RealVis11','9aba3090-b462-4868-9531-eb7249de2e24'),
        'Deliberate':('Deliberate','b559ba3b-ee88-4ecd-8737-8d622bd06b85'),
        'GeFeMi':('GeFeMi_1-1','655abe54-83f6-49c6-866d-6fb5c63c8e7c'),
        'Hassan Blend 1.4': ('hassanblend/hassanblend1.4','60867458-2579-4f38-b5e2-ed2d9178f836'),
        'Hassan Blend 1.5':('hassanblend/HassanBlend1.5','0ba7604d-5068-4d1b-87cb-3da68fda0f27'),
        }
    replicate_keys = {
        'Prompt Parrot': ("kyrick/prompt-parrot","7349c6ce7eb83fc9bc2e98e505a55ee28d7a8f4aa5fb6f711fad18724b4f2389"), 
        'Cog Prompt Parrot': ('2feet6inches/cog-prompt-parrot', 'b08782b399f76e0671f48e564cf5efc9a89941ed12399b8deae21b04abbb7b4c'), 
        'Img2Prompt': ('methexis-inc/img2prompt', '50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5'), 
        'Clip Interrogator': ('pharmapsychotic/clip-interrogator', 'a4a8bafd6089e1716b06057c42b19378250d008b80fe87caa5cd36d40c1eda90'),
        'Real-ESRGAN': ("nightmareai/real-esrgan",'42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b'), 
        'SwinIR Image Restoration': ("jingyunliang/swinir",'660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a'), 
        'Codeformer Face Restoration': ("sczhou/codeformer",'7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56'), 
        'Latent-SR Upscaler': ("nightmareai/latent-sr",'9117a98dd15e931011b8b960963a2dec20ab493c6c0d3a134525273da1616abc'), 
        'GFPGAN Face Restoration': ("tencentarc/gfpgan",'9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3'),
        }
    st.sidebar.markdown("**Settings**")
    admin_code = st.sidebar.text_input("Admin Code (Optional)", type="password", key="admin")
    scheduler = st.sidebar.selectbox('Select Sampling Method',("DPMSolverMultistepScheduler", "LMSDiscreteScheduler", "DDIMScheduler", "PNDMScheduler", "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler"), index=0)
    steps = st.sidebar.slider("Number of Inference Steps", min_value=1, max_value=150, value=20)
    num_images = st.sidebar.slider("Number of Images Per Prompt", min_value=1, max_value=4, value=1, step=1)
    scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=30.0, value=7.0, step=0.5)
    height = st.sidebar.slider("Image Height", min_value=64, max_value=1024, value=512, step=64)
    width = st.sidebar.slider("Image Width", min_value=64, max_value=1024, value=512, step=64)
    seed_option = st.sidebar.radio('Select Seed Option', ('Random Seed', 'Manual Seed'), index=0)
    if seed_option == 'Manual Seed':
        seed = st.sidebar.number_input('Enter Manual Seed', min_value=None, max_value=None, step=1)
    if seed_option == 'Random Seed':
        seed = random.randint(1000000000,9999999999)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Text2Image", "Image2Image", "Prompt Engineering", "Post Processing", 'PNG Info'])
    with tab1:
        st.header("Text2Image")
        if admin_code==st.secrets["admin_code"]:
            model = st.selectbox('Select Diffusion Model',('Stable 1.5', 'Stable 2.1', 'OpenJourney V2', 'WikiArt V2', 'Stable Analog', 'portrait+ style', 'Dreamlike Photoreal 2.0','Photorealistic Fuen V1','classicnegative photo','Protogen V2.2','Protogen V3.4','Protogen V5.3','Protogen V5.8','f222','Realistic Vision V1.1','Deliberate','GeFeMi','Hassan Blend 1.4','Hassan Blend 1.5'), index=1)
        else:
            model = st.selectbox('Select Diffusion Model',('Stable 1.5', 'Stable 2.1', 'OpenJourney V2', 'WikiArt V2', 'Stable Analog', 'portrait+ style', 'Dreamlike Photoreal 2.0','Photorealistic Fuen V1','classicnegative photo','Protogen V2.2'), index=1)
        if model=='Stable 1.5': 
            txt2img_prompt = st.text_input("Text to Image Prompt", key="txt2imgP")
        if model!='Stable 1.5':
            col1, col2 = st.columns(2)
            with col1:
                txt2img_prompt = st.text_area("Prompt", "")
            with col2:
                txt2img_negative_prompt = st.text_area("Negative Prompt", "")
        sub_col, download_col = st.columns(2)
        with sub_col:
            generate = st.button('Generate Image', key="tab1")
        status = st.empty()
        canvas = st.empty()
        if generate:
            with st.spinner("Loading..."):
                if 'Stable 2.1' in model:
                    safety_filter = 'false'
                else:
                    safety_filter = False
                if model=='GeFeMi' or model=='Realistic Vision V1.1' or model=="Protogen V5.8" or model=="Protogen V5.3" or model=="Protogen V3.4" or model=='Deliberate':
                    model_inputs = {
                        'prompt': txt2img_prompt,
                        'height': height,
                        'width': width,
                        'num_inference_steps': steps,
                        'guidance_scale': scale,
                        'negative': txt2img_negative_prompt,
                        'num_images_per_prompt': 1,
                        'seed': seed,
                    }
                    status.info('Processing request...', icon="⏳")
                    out = banana.run(api_key, model_keys[model][1], model_inputs)
                    model_inputs['model'] = model
                    model_inputs['scheduler'] = 'EulerAncestralDiscreteScheduler'
                    decode_and_save1(out["modelOutputs"][0]["image_base64"], model_inputs)

                elif model=='Stable 1.5':
                    modelInputs = {
                        "prompt": txt2img_prompt,
                        "num_inference_steps":steps,
                        "num_images_per_prompt": num_images,
                        "guidance_scale":scale,
                        "height":height,
                        "width":width,
                        "seed":seed,
                    }
                    callInputs = {
                        "MODEL_ID": model_keys[model][0],
                        "PIPELINE": "StableDiffusionPipeline",
                        "SCHEDULER": scheduler,
                        "safety_checker": safety_filter,
                    }
                    status.info('Processing request...', icon="⏳")
                    out = banana.run(api_key, model_keys[model][1], { "modelInputs": modelInputs, "callInputs": callInputs })
                    merge_inputs(callInputs, modelInputs)
                    if modelInputs['num_images_per_prompt'] < 2:
                        decode_and_save1(out["modelOutputs"][0]["image_base64"], modelInputs)
                    else:        
                        decode_and_save_multi(out["modelOutputs"][0]["images_base64"], modelInputs)

                else:
                    modelInputs = {
                        "prompt": txt2img_prompt,
                        "negative_prompt": txt2img_negative_prompt,
                        "num_inference_steps":steps,
                        "num_images_per_prompt": num_images,
                        "guidance_scale":scale,
                        "height":height,
                        "width":width,
                        "seed":seed,
                    }
                    callInputs = {
                        "MODEL_ID": model_keys[model][0],
                        "PIPELINE": "StableDiffusionPipeline",
                        "SCHEDULER": scheduler,
                        "safety_checker": safety_filter,
                    }
                    status.info('Processing request...', icon="⏳")
                    out = banana.run(api_key, model_keys[model][1], { "modelInputs": modelInputs, "callInputs": callInputs })
                    merge_inputs(callInputs, modelInputs)
                    if modelInputs['num_images_per_prompt'] < 2:
                        decode_and_save1(out["modelOutputs"][0]["image_base64"], modelInputs)
                    else:        
                        decode_and_save_multi(out["modelOutputs"][0]["images_base64"], modelInputs)
    with tab2:
        st.header("Image2Image")
        if admin_code==st.secrets["admin_code"]:
            model = st.selectbox('Select Diffusion Model',('Stable 1.5', 'Stable 2.1', 'OpenJourney V2', 'WikiArt V2', 'Stable Analog', 'portrait+ style', 'Dreamlike Photoreal 2.0','Photorealistic Fuen V1','classicnegative photo','Protogen V2.2','f222','Hassan Blend 1.4','Hassan Blend 1.5'), index=1, key="img2img_model")
        else:
            model = st.selectbox('Select Diffusion Model',('Stable 1.5', 'Stable 2.1', 'OpenJourney V2', 'WikiArt V2', 'Stable Analog', 'portrait+ style', 'Dreamlike Photoreal 2.0','Photorealistic Fuen V1','classicnegative photo','Protogen V2.2'), index=1, key='img2img_model')
        url_toggle = st.checkbox('Enable Image File Upload')
        if url_toggle:
            uploaded_file = st.file_uploader("Upload Image File", type=['png','jpeg','jpg'], key="img2img")
            if uploaded_file is not None:
                init_image = PIL.Image.open(uploaded_file)
                init_image.save('init_image.png')
        else:
            image_url = st.text_input("Image URL", key="img2img_URL")
        if model=='Stable 1.5': 
            img2img_prompt = st.text_input("Image to Image Prompt", key="img2imgP")
        if model!='Stable 1.5':
            col1, col2 = st.columns(2)
            with col1:
                img2img_prompt = st.text_area("Image to Image Prompt", "", key="img2imgP2")
            with col2:
                img2img_negative_prompt = st.text_area("Enter Negative Prompt", key="img2imgN")
        strength = st.slider('Transformation strength', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        sub_col, download_col = st.columns(2)
        with sub_col:
            generate = st.button('Generate Image', key="tab2")
        status = st.empty()
        canvas = st.empty()
        if generate:
            with st.spinner("Loading..."):
                if 'Stable 2.1' in model:
                    safety_filter = 'false'
                else:
                    safety_filter = False
                if url_toggle==False:
                    urllib.request.urlretrieve(image_url, 'init_image.png')
                if model=='Stable 1.5':
                    modelInputs = {
                        "prompt": img2img_prompt,
                        "num_inference_steps":steps,
                        "num_images_per_prompt": num_images,
                        "guidance_scale":scale,
                        "height":height,
                        "width":width,
                        "seed":seed,
                        "strength": strength,
                        "init_image": b64encode_file('init_image.png'),
                    }
                    callInputs = {
                        "MODEL_ID": model_keys[model][0],
                        "PIPELINE": "StableDiffusionImg2ImgPipeline",
                        "SCHEDULER": scheduler,
                        "safety_checker": safety_filter,
                    }
                else:
                    modelInputs = {
                        "prompt": img2img_prompt,
                        "negative_prompt": img2img_negative_prompt,
                        "num_inference_steps":steps,
                        "num_images_per_prompt": num_images,
                        "guidance_scale":scale,
                        "height":height,
                        "width":width,
                        "seed":seed,
                        "strength": strength,
                        "init_image": b64encode_file('init_image.png'),
                    }
                    callInputs = {
                        "MODEL_ID": model_keys[model][0],
                        "PIPELINE": "StableDiffusionImg2ImgPipeline",
                        "SCHEDULER": scheduler,
                        "safety_checker": safety_filter,
                    }
                status.info('Processing request...', icon="⏳")
                out = banana.run(api_key, model_keys[model][1], { "modelInputs": modelInputs, "callInputs": callInputs })
                merge_inputs(callInputs, modelInputs)
                if modelInputs['num_images_per_prompt'] < 2:
                    decode_and_save1(out["modelOutputs"][0]["image_base64"], modelInputs)
                else:        
                    decode_and_save_multi(out["modelOutputs"][0]["images_base64"], modelInputs)
    with tab3:
        st.header("Prompt Engineering")
        choice = st.selectbox('Select Prompt Engineering Tool',('Prompt Parrot', 'Cog Prompt Parrot', 'Img2Prompt', 'Clip Interrogator'), index=0)
        replicate_model = replicate.models.get(replicate_keys[choice][0])
        model_version = replicate_model.versions.get(replicate_keys[choice][1])
        if 'Parrot' in choice:
            text2prompt = st.text_input("Text to Prompt", key="txt2P")
        else:
            toggle = st.checkbox('Enable Image Upload')
            if toggle:
                img2prompt_upload = st.file_uploader("Enable Image File Upload", type=['png','jpeg','jpg'], key="img2prompt")
                if img2prompt_upload is not None:
                    img2prompt_file = PIL.Image.open(img2prompt_upload)
                    img2prompt_file.save('img2prompt.png')
                    img2prompt = pathlib.Path("img2prompt.png")
            else:
                img2prompt = st.text_input("Image URL", key="txt2P_URL")
        generate_prompt = st.button('Generate Prompt', key="tab3")
        status = st.empty()
        prompts = st.empty()
        if generate_prompt:
            with st.spinner("Loading..."):
                status.info('Processing request...', icon="⏳")
                if 'Parrot' in choice:
                    output = model_version.predict(prompt=text2prompt)
                    status.success('Successfully generted prompts', icon="✅")
                    prompts.text(output)
                else:
                    output = model_version.predict(image=img2prompt)
                    status.success('Successfully generted prompt', icon="✅")
                    prompts.write(output)
    with tab4:
        st.header('Image Post Processing Lab')
        options = st.selectbox('Select Image Post Processing Tool', ('Real-ESRGAN', 'SwinIR Image Restoration', 'Codeformer Face Restoration', 'GFPGAN Face Restoration'), index=0)
        replicate_model = replicate.models.get(replicate_keys[options][0])
        model_version = replicate_model.versions.get(replicate_keys[options][1])
        img_lab_uploader = st.file_uploader("Upload Image File", type=['png','jpeg','jpg'], key="img_lab")
        if img_lab_uploader is not None:
            img_lab_file = PIL.Image.open(img_lab_uploader)
            img_lab_file.save('img_lab.png')
            img_lab = pathlib.Path("img_lab.png")
        if 'Real-ESRGAN' in options:
            face_resto = st.checkbox('Enable Face Enhance')
            scale_setting = st.slider("Factor to scale image by (maximum: 10)", min_value=0.0, max_value=10.0, value=3.0, step=0.01)
        if 'SwinIR Image Restoration' in options:
            noise_setting = 15
            jpeg_compression = 40
            task = st.radio('Choose a task',('Real-World Image Super-Resolution-Large', 'Real-World Image Super-Resolution-Medium', 'Grayscale Image Denoising', 'Color Image Denoising', 'JPEG Compression Artifact Reduction'), index=0)
            if "Denoising" in task:
                noise_setting = st.radio('Noise level, activated for Grayscale Image Denoising and Color Image Denoising. Leave it as default or arbitrary if other tasks are selected', ('15','25','50'), index=0)
        if 'Codeformer Face Restoration' in options:
            fidelity = st.slider('Balance the quality (lower number) and fidelity (higher number). (maximum: 1)', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
            background = st.checkbox('Enhance background image with Real-ESRGAN')
            face_up = st.checkbox('Upsample restored faces for high-resolution AI-created images')
            upscaler = st.slider('The final upsampling scale of the image', min_value=0, max_value=10, value=2, step=1)
        if 'GFPGAN Face Restoration' in options:
            ver = st.radio('GFPGAN version. v1.3: better quality. v1.4: more details and better identity.', ('v1.2', 'v1.3', 'v1.4', 'RestoreFormer'), index=2)
            rescaling_factor = st.slider('Rescaling factor', min_value=0.0, max_value=2.0, value=1.5, step=0.01)
        sub_col, download_col = st.columns(2)
        with sub_col:
            use_tool = st.button('Submit', key='tab4')
        status = st.empty()
        canvas = st.expander("Show Image")
        if use_tool:
            with st.spinner("Loading..."):
                status.info('Processing request...', icon="⏳")
                image_filename = (f'{img_lab_uploader.name}_{options}.png')
                if 'Real-ESRGAN' in options:
                    output = model_version.predict(image=img_lab, scale=scale_setting, face_enhance=face_resto)
                    img_lab_download(output, image_filename)
                if 'SwinIR Image Restoration' in options:
                    output = model_version.predict(image=img_lab, task_type=task, noise=int(noise_setting), jpeg=jpeg_compression)
                    img_lab_download(output, image_filename)
                if 'Codeformer Face Restoration' in options:
                    output = model_version.predict(image=img_lab, codeformer_fidelity=fidelity, background_enhance=background, upscale=upscaler)
                    img_lab_download(output, image_filename)
                if 'GFPGAN Face Restoration' in options:
                    output = model_version.predict(img=img_lab, version=ver, scale=rescaling_factor)
                    img_lab_download(output, image_filename)
    with tab5:
        uploaded_file = st.file_uploader("Choose an image...", type=["png"], key="png")
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            image_info = pil_image.info
            st.json(image_info['parameters'])
