from diffusers import StableDiffusionPipeline
import torch 
import sys 

def main(text : str):
    fn = "{0}.png".format("+".join(text.split(" ")))
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("mps")
    pipe.enable_attention_slicing()
    result = pipe(text)
    image = result.images[0]
    image.save(fn) 
    print(fn)

if len(sys.argv) >= 2:
    main(" ".join(sys.argv[1:]))
else:
    print("enter prompt")