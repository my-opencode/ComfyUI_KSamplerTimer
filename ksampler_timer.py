import os
import sys
import time

# Get the absolute path of various directories
# my_dir = os.path.dirname(os.path.abspath(__file__))
# custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '..'))
# comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))
custom_nodes_dir = os.path.dirname(os.path.abspath(__file__))
comfy_dir = os.path.abspath(os.path.join(custom_nodes_dir, '..', '..'))

# Append comfy_dir to sys.path & import files
sys.path.append(comfy_dir)
import comfy.samplers
from nodes import common_ksampler
sys.path.remove(comfy_dir)

class KSamplerTimer:
    """
    A custom ksampler node that wraps the default ksampler with a timer
    Outputs the latent image and the generation time in seconds as a string and as a float.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT", "STRING", "FLOAT")
    RETURN_NAMES = ("LATENT", "GENERATION_TIME_STRING", "GENERATION_TIME")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        execution_start_time = time.perf_counter()
        out =  common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        current_time = time.perf_counter()
        execution_time = current_time - execution_start_time
        return (out[0], "{:.2f}".format(execution_time), execution_time)

    def test(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, print_to_screen="disable"):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                seed: {seed}
                steps: {steps}
                cfg: {cfg}
                sampler_name: {sampler_name}
                scheduler: {scheduler}
                positive: {positive}
                negative: {negative}
                denoise: {denoise}
            """)
        return (latent_image, "done")


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "KSamplerTimer": KSamplerTimer
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerTimer": "KSampler (timer)"
}
