#
# Written by Nguyen Minh Hieu (Charlie), 2024
#
import numpy as np
from pysdf import SDF
import igl
import json

def normalize_unit_cube(v):
    bmin = np.min(v, axis=0)
    bmax = np.max(v, axis=0)
    c = (bmin + bmax) / 2
    s = np.max(bmax - bmin)
    return (v - c) / s

def main(filename="bunny"):
    v,f = igl.read_triangle_mesh(f"./{filename}.obj")
    v = normalize_unit_cube(v)

    res = 128
    b = np.linspace(-1,1,res)
    p = np.stack(np.meshgrid(b,b,b),axis=-1).reshape(-1,3)

    sdf = SDF(v,f)
    d = (-sdf(p).reshape(res,res,res) + 2.0) / 4.0
    d = (d * 255).astype(np.uint8)

    # import matplotlib.pyplot as plt

    # plt.imshow(d[:,:,32])
    # plt.show()

    d_serialized = d.T.reshape(-1).tolist()

    json_content = {
        "dimX": res,
        "dimY": res,
        "dimZ": res,
        "data": d_serialized
    }

    with open(f"{filename}.json","w") as f:
        json.dump(json_content, f)

main()
