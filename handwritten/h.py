import torch
from diffusers import StableDiffusionPipeline, PNDMScheduler
from flask import Flask, render_template, request
import os

def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to("cpu")
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    return pipe

app = Flask(__name__)
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text_prompt = request.form['prompt']
    image = model(text_prompt, num_inference_steps=50).images[0]
    image_path = "static/generated_image.png"
    image.save(image_path)
    return render_template('index.html', image_path=image_path)

if __name__ == '__main__':
    app.run(debug=False)