### README -- artery tracing 4x (at4x)

0. You need to run this on Linux and data should be in local drive (e.g. d or e).

1. Open MobaXterm and doubli-click WSL-Ubuntu (or something equivalent).

2. (one time) create conda environment from a yml file (env_at4x.ylm)
(base)$ conda env create --name at4x --file=env_at4x.yml

2. Activate the created conda environment
(base)$ conda activate at4x

3. Start Jupyter Notebook
(at4x)$ jupyter notebook

4. Navigate to the folder and open 'RUN_quantify_4x_artery_data.ipynb'

5. Specify the data folder that you want to process

6. Run all cells

