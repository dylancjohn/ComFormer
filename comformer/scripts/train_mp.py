import os

from comformer.train_props import train_prop_model

props = [
    "e_form",
    "gap pbe",
    "bulk modulus",
    "shear modulus",
]

OUTPUT_DIR = "/home/dylan/ComFormer/outputs/icomformer_mp_bandgap"

# iComFormer, Materials Project band gap ("gap pbe"), matching the ICLR 2024 paper's
# reported setting (Table 2 / 12 and Appendix A.6.1): conv_layers=4, edge_layers=1
# (iComformerConfig defaults), 25 neighbors, cutoff 4.0, L1 loss, Adam + polynomial
# LR schedule (0.0005 -> 0.00001), 500 epochs.
#
# save_dataloader=True + an absolute file_name cache the built k=25 crystal graphs
# for all 69k MP structures (the expensive, hour-plus preprocessing step) under
# OUTPUT_DIR, so a restart/resume reloads them instead of rebuilding from scratch.
# (comformer/data.py checks these paths relative to the cwd, not output_dir, hence
# the absolute path here.)
train_prop_model(
    learning_rate=0.0005,
    criterion="l1",
    name="iComformer",
    dataset="megnet",
    prop=props[1],
    pyg_input=True,
    n_epochs=500,
    max_neighbors=25,
    cutoff=4.0,
    batch_size=64,
    use_lattice=True,
    output_dir=OUTPUT_DIR,
    use_angle=False,
    scheduler="polynomial",
    save_dataloader=True,
    file_name=os.path.join(OUTPUT_DIR, "graph_cache"),
)
