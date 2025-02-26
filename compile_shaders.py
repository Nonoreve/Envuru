#!/usr/bin/python3
import subprocess
from sys import argv

glsc_location = "/home/nonoreve/Stuff/VulkanSDK/x86_64/bin/glslc"
required_shader_pairs = ["texture"]
required_solo_shaders = []

for pair in required_shader_pairs:
    subprocess.run(f"{glsc_location} {argv[1]}/{pair}.vert -o {argv[2]}/vert.spv", shell=True)
    subprocess.run(f"{glsc_location} {argv[1]}/{pair}.frag -o {argv[2]}/frag.spv", shell=True)
