#!/usr/bin/python3
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("shaders_location")
parser.add_argument("spv_output")
parser.add_argument("--required_shader_pairs", action='extend', nargs='+')
args = parser.parse_args()

glsc_location = "/home/nonoreve/Stuff/VulkanSDK/x86_64/bin/glslc"

for pair in args.required_shader_pairs:
    subprocess.run(f"{glsc_location} {args.shaders_location}/{pair}.vert -o {args.spv_output}/vert.spv", shell=True)
    subprocess.run(f"{glsc_location} {args.shaders_location}/{pair}.frag -o {args.spv_output}/frag.spv", shell=True)
