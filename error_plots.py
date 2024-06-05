import safetensors
import torch
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
import seaborn as sns
import numpy as np
import glob
from tqdm import tqdm
sns.set_theme(style="whitegrid", context="talk")

def quant_tensors(w, w_bit=4, G=128):
    w = w.view(-1, G).half()
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**w_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    scale_zeros = zeros * scales
    w_quant = torch.round((w + scale_zeros) / scales)
    w_dequant = (w_quant * scales) - scale_zeros
    quant_err = torch.abs(w - w_dequant)
    return quant_err, scales, zeros

def rounding_err_hist(quant_err, name=""):
    bins = np.logspace(np.log10(1e-8), np.log10(quant_err.amax() + 1e-1), 100)
    plt.hist(quant_err.flatten(), bins=bins, edgecolor="none", log=True, color='blue')
    plt.xscale("log")
    plt.xlabel("Weight rounding error")
    plt.ylabel("Number of weights")
    plt.title("Histogram of weight rounding error after activation scaling")
    plt.suptitle(name)

def joint_scale_err_plot(quant_err, scales, name=""):
    grid = sns.JointGrid()
    xbins = np.logspace(np.log10(1e-5), np.log10(quant_err.amax() + 1e-1), 100)
    ybins = np.logspace(np.log10(1e-5), np.log10(scales.amax() + 1e-1), 100)
    grid.ax_joint.hist2d(quant_err.amax(dim=1).flatten(), scales.flatten(), bins=(xbins, ybins), norm=mpl.colors.LogNorm(), cmap='viridis')
    grid.ax_marg_x.hist(quant_err.amax(dim=1).flatten(), bins=xbins, edgecolor="none", log=True, color='blue')
    grid.ax_marg_y.hist(scales.amax(dim=1).flatten(), bins=ybins, edgecolor="none", log=True, color='blue', orientation='horizontal')
    grid.ax_joint.set_xscale("log")
    grid.ax_joint.set_yscale("log")
    grid.ax_joint.set_xlabel("Quantization group scale")
    grid.ax_joint.set_ylabel("Maximum weight rounding error in group")
    grid.ax_marg_x.set_title("Histogram of maximum weight rounding error and group scale")
    grid.figure.colorbar(grid.ax_joint.collections[0], ax=grid.ax_marg_y)
    grid.figure.suptitle(name, y=1.04)

def main():
    parser = argparse.ArgumentParser(description="Plot weight quantization error")
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--model-folder", type=str)

    args = parser.parse_args()
    for file in glob.glob(f"{args.model_folder}/*.safetensors"):
        st = safetensors.safe_open(file, "pt")
        parts = ["down", "up", "gate"]
        layers = [k for k in st.keys() if any([p in k for p in parts])]
        for layer_name in tqdm(layers):
            w = st.get_tensor(layer_name)
            quant_err, scales, zeros = quant_tensors(w)
            rounding_err_hist(quant_err, name=layer_name)
            plt.savefig(f"{args.outdir}/{layer_name}_rounding_err_hist.png", bbox_inches='tight')
            plt.close()
            joint_scale_err_plot(quant_err, scales, name=layer_name)
            plt.savefig(f"{args.outdir}/{layer_name}_joint_scale_err_plot.png", bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    main()