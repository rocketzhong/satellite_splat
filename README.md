# SatelliteSplat
Based on splatfacto

## Install and Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd satellite_splat/
pip install -e .
ns-install-cli
```

## Running the new method
This repository is the official implements of "satellite_splat". "https://arxiv.org/......还没写出来"

To train with it, run the command:
```
ns-train satellite_splat --data [PATH]
```