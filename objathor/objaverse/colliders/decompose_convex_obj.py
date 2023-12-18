import pdb
import sys

file_name = sys.argv[1]

with open(file_name, "r") as f:
    lines = [l for l in f]

decomposed_files = []

current_file = []
for l in lines:
    if l[:2] == "o ":
        if len(current_file) > 0:
            decomposed_files.append(current_file)
        current_file = []

    current_file.append(l)
if len(current_file) > 0:
    decomposed_files.append(current_file)

vertices_so_far = 0
for i in range(len(decomposed_files)):
    current_file = decomposed_files[i]
    file_to_write = file_name.replace(".obj", f"_{i}.obj")
    with open(file_to_write, "w") as f:
        for l in current_file:
            if l[:2] == "f ":
                line_to_write = l.replace("\n", "").split(" ")[1:]
                # print(f"line: {line_to_write}")
                vertex_numbers = [int(x) - vertices_so_far for x in line_to_write if x != ""]
                assert len(vertex_numbers) == 3
                l = f"f {vertex_numbers[0]} {vertex_numbers[1]} {vertex_numbers[2]}\n"
            f.write(l)

    current_vertices = len([l for l in current_file if l[:2] == "v "])
    vertices_so_far += current_vertices
