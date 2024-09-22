# Attention Map-based Inappropriate Degeneration with Stable Diffusion

ABSD is the post-hoc method for degenerate inappropriateness in output of [Stable Dffusion](https://github.com/CompVis/stable-diffusion) v1.4 .

ABSD manipulat cross-attention in Unet of stable diffusion scheduler.

Calculate and utilize the difference between safe prompt(ex `nudity`) attention maps and text-prompt attentiom maps.

ABSD is a generative model that applies methods to reduce the inadequacy of the existing stable diffusion model for generating images.

Our code can take a text prompt as input and generate an image with reduced inappropriateness. 

Allows users to enter inappropriateness they wish to reduce with text input of their choice (`--safe_prompt`). 

* Default safe prompts is `i2p` (`hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty`)
* To generate images, using constom prompt (`--prompts` in `test.py`) with `--safe prompt`
* If you want to generate an image with a prompt of your choice, you can use `test.py` to generate one image corresponding to the prompt.
* If you want to create a variety of images, you can use `generate-images.py`, which uses a CSV file to generate images, to create images for different prompts.

##  Quick start

We recommend running it in a virtual environment using a Docker image

### Requirement

* Our architecture is implemented Python 3.8.13, Pytorch 1.13.0a0+08820cb, Linux 20.04.2 and CUDA 11.4
* Install requirements with `docker/requirements.txt`.

```
sudo git clone https://github.com/Seoultech-IXLAB/ABSD.git
cd ABSD
pip install Python==3.8.13, Pytorch==1.13.0a0+08820cb
pip install -r docker/requirementes.txt
python test.py --model_name='SD-v1-4' --prompts 'extreme close up, portrait style, sideways teeth, horror, man, eyes wide open, pain, blood drip from eyeballs, colours, dark creature in background ' --save_path 'test' --num_samples 1 --ddim_steps 50 --method "attention" --safe_steps 0 --lamda 0.3
# generated images save in test/
cd test/
```

### Use Docker

```
sudo git clone https://github.com/Seoultech-IXLAB/ABSD.git
cd /ABSD/docker
```
Edit docker-compose.yml volume to your path 
(ex, `/change/your/path` to `~/workspace/ABSD:/workspace/ABSD`)
```
sudo docker-compose up -d
sudo docker exec -it ABSD bash
## In docekr bashshell
python test.py --model_name='SD-v1-4' --prompts 'extreme close up, portrait style, sideways teeth, horror, man, eyes wide open, pain, blood drip from eyeballs, colours, dark creature in background ' --save_path 'test' --num_samples 1 --ddim_steps 50 --method "attention" --safe_steps 0 --lamda 0.3
# generated images save in test/
cd test/
```
If you tried to create it, you'll see three different images in the /test folder based on the latent, prompt, and attention edits. 


## Generating Images

### Generate custom prompt images

if you generate images with your prompts, follow this instructions:
```
#in ABSD/
python test.py --model_name='SD-v1-4' --prompts 'your prompts' --save_path 'test' --num_samples 1 --safe_steps 0 --lamda 0.3

```
generated image save in safe_path (`test/`)

#### Required Arguments

* `model_name` : select version of stable diffusion (`SD-v1-4`, `SD-V2`).
* `prompts` : input your text prompts (i.e, `hello world`).
* `save_path` : path to save generated images (i.e, `result`-> `result/`).
* `safe_steps` : Determine the time stemp to calculate safe guidance.
* `lamda` : Determine the scaling value for attention map of residual map (inappr - prompt)
* `device` : If you don't have a GPU, enter `cpu`. If you do have a GPU, you don't need to use it

* 
### Generate images with CSV

I2P Benchmark Source Repository: https://github.com/ml-research/i2p

I2P benchmark dataset : https://huggingface.co/datasets/AIML-TUDA/i2p

Other dataset CSV files : https://github.com/rohitgandikota/erasing

To generate images from one of the our methods use the following instructions:

* To use `generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`).
* To generate multiple images per prompt use the argument `num_samples`. It is default to 1.
* Set diffusion scheduler step with `--ddip_steps`. It is default 50.

```
CUDA_VISIBLE_DEVICES=0 python generate-images.py --model_name "SD-v1.4" --prompts_path 'data/unsafe-prompts4703.csv' --save_path 'result' --safe_steps 0 --lamda 0.3

```

generated images save in `{save_path}_i2p_{safe_steps}_{lamda}/`.

#### Required Arguments

* `model_name` : Select version of stable diffusion (`SD-v1-4`, `SD-V2`).
* `prompts_path` : Path of a CSV file provide test prompts, The CSV files are located in the data/ folder  (i.e, `data/unsafe_prompts4703.csv`).
* `save_path` : Path to save generated images.
* `safe_steps` : Determine the time stemp to calculate safe guidance.
* `lamda` : Determine the scaling value for attention map of residual map (inappr - prompt)
* `device` : If you don't have a GPU, enter `cpu`. If you do have a GPU, you don't need to use it

