from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from src.generate_step import crossattention_edit_latent_step
import torch
from PIL import Image
import pandas as pd
import argparse
import os
import gc


def generate_images(model_name, prompts, save_path , safe_step, lamda, safe_prompt="i2p", method="attention", device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=50, num_samples=1, from_case=0 , learning_late = 0.01, test=False):

    '''
    Function to generate images from diffusers code
    
    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    safe_step : int
        The inappropriate degeneration starting step.
    lamda : float
        scaling value for inappropriate degeneration

    safe_prompt : str, optional
        prompt for degenerating inappropriateness
    method : str, optional
        choice inappropriate degeneration method.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 50.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.
    learning_late : float, optional
        learning late for optimizing latent or prompt using SGD. 

    Returns
    -------
    None.

    '''
    if model_name == 'SD-v1-4':
        dir_ = "CompVis/stable-diffusion-v1-4"
    elif model_name == 'SD-V2':
        dir_ = "stabilityai/stable-diffusion-2-base"
    else:
        dir_ = "CompVis/stable-diffusion-v1-4" # all the erasure models built on SDv1-4
        
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")

    if 'SD' not in model_name:
        try:
            model_path = f'models/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
            unet.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f'Model path is not valid, please check the file name and structure: {e}')
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device


    folder_path = f'{save_path}'
    os.makedirs(folder_path, exist_ok=True)

    prompt = [str(prompts)]*num_samples
    #seed = row.evaluation_seed
    #case_number = row.case_number
    
    height = image_size                        # default height of Stable Diffusion
    width = image_size                         # default width of Stable Diffusion

    num_inference_steps = ddim_steps           # Number of denoising steps

    generator = torch.manual_seed(1)        # Seed generator to create the inital latent noise

    batch_size = len(prompt)
    if 'i2p' in safe_prompt:
        i2p = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty"
        safe_input = tokenizer(i2p, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    else:
        safe_input = tokenizer(safe_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    safe_embeddings = text_encoder(safe_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings, safe_embeddings])

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    if "attention" in method:   
        latents = crossattention_edit_latent_step(safe_step, lamda, latents,text_embeddings, unet, scheduler,device, guidance_scale)
   
    else:
        print("method not found try again")
        assert True

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    for num, im in enumerate(pil_images):
        im.save(f"{folder_path}/{model_name}_{safe_step}_{lamda}_test_{num}.png")
    print("end")


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--prompts', help='generate image with your prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--safe_steps', help='The inappropriate degeneration starting step', type=int, required=False, default=0)
    parser.add_argument('--lamda', help="scaling", type=float, required=True, default=0.3)


    parser.add_argument('--safe_prompt', help= 'prompt for degenerating inappropriateness', type=str, required=False, default="i2p") 
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--learning_late', help='learning late to run eval', type=float, required=False, default=0.1)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)  
    parser.add_argument('--method', help="method of degenerate inappropliate", type=str, required=False, default="attention")
    parser.add_argument("--test", type=bool ,required=False, default=False)
    
    args = parser.parse_args()

    model_name = args.model_name
    prompts= args.prompts
    safe_prompt =args.safe_prompt
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    from_case = args.from_case
    num_samples= args.num_samples
    ddim_steps = args.ddim_steps
    learning_late = args.learning_late
    method = args.method
    test = args.test

    # Hyper-parameters
    safe_step = args.safe_steps
    lamda = args.lamda    

    save_path = f"{args.save_path}"
    print(f"save path : {save_path}")

    
    generate_images(model_name, prompts, save_path, safe_step, lamda, safe_prompt=safe_prompt, method=method, device=device,
                    guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case, learning_late = learning_late, test=test)


