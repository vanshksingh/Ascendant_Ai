import argparse
import re
from generator import image_generator

if __name__ == '__main__':

    generator = image_generator(
        base_filename="",
        amount=1,
        prompt= "a person with a hat, in a forest, at night, with a lantern",
        prompt_size= 10,
        negative_prompt= "nudity text",
        style="cinematic",
        resolution="512x768",
        guidance_scale=7
    )

    for _ in generator:
        pass
