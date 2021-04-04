# Accelerating Antimicrobial Discovery with Controllable Deep Generative Models and Molecular Dynamics
### This work will be published in _Nature Biomedical Engineering_ on March 11, 2021
### URL :  https://www.nature.com/articles/s41551-021-00689-x 

> De novo therapeutic design is challenged by a vast chemical repertoire and multiple constraints, e.g.,
>  high broad-spectrum potency and low toxicity. This project proposes CLaSS (Controlled Latent attribute 
Space Sampling) - an efficient computational method for attribute-controlled generation of molecules, which leverages 
guidance from classifiers trained on an informative latent space of molecules modeled using a deep generative autoencoder. 
We screen the generated molecules for additional key attributes by using deep learning classifiers in conjunction with novel 
features derived from atomistic simulations.



## Setup 
+ The `amp_gen.yml` lists are the required dependencies for the project.
+ Use `amp_gen.yml` to create your own conda environment to run this project. Command: `conda-env create -f amp_gen.yml`

## Usage

### Phase 1: Autoencoder (VAE/WAE) Training
+ `./run.sh`. This will run with default config from `cfg.py`. Since `cfg.runname=default` the output goes to `output/default` and `tb/default`.
+ `python main.py --tiny 1` for fast testing with default config file.
+ Additionally, one could explicitly run the individual scripts as follows:
  * > python main.py --phase 1
  * > python static_eval.py --config_json output/dir/config_overrides.json

### Phase 2: CLaSS (Controlled Latent attribute Space Sampling)
+ > python sample_pipeline.py --config_json output/default/config_overrides.json --samples_outfn_prefix samples --Q_select_amppos 0

### Data: 
+ `data_processing/data` dir has the short versions of data files required by our data curation code `data_processing/create_datasets.py`
+ For the full version of dataset use following links to download full version of data files that are publicly available. 
+ UNIPROT: [https://www.uniprot.org/uniprot/?query=reviewed:yes] and [https://www.uniprot.org/uniprot/?query=reviewed:no]
+ SATPDB: [http://crdd.osdd.net/raghava/satpdb/]
+ DBAASP: [https://dbaasp.org]
+ AMPEP: [https://cbbio.cis.um.edu.mo/software/AmPEP/]
+ ToxinPred: [https://webs.iiitd.edu.in/raghava/toxinpred/dataset.php]

## Related Visualization Tools
+ Peptide Walker : [https://peptide-walk.mybluemix.net](https://peptide-walk.mybluemix.net)
+ Cogmol Drug Exploration: [https://covid19-mol.mybluemix.net](https://covid19-mol.mybluemix.net)

## Citations

Please cite the following articles:

```
@article{das2020accelerating,
  title={Accelerating Antimicrobial Discovery with Controllable Deep Generative Models and Molecular Dynamics},
  author={Das, Payel and Sercu, Tom and Wadhawan, Kahini and Padhi, Inkit and Gehrmann, Sebastian and Cipcigan, Flaviu and Chenthamarakshan, Vijil and Strobelt, Hendrik and Santos, Cicero dos and Chen, Pin-Yu and others},
  journal={arXiv preprint arXiv:2005.11248},
  year={2020}
}
```



```
@article{chenthamarakshan2020cogmol,
  title={CogMol: Target-specific and selective drug design for COVID-19 using deep generative models},
  author={Chenthamarakshan, Vijil and Das, Payel and Hoffman, Samuel C and Strobelt, Hendrik and Padhi, Inkit and Lim, KW and others},
  journal={arXiv: 2004.01215},
  year={2020}
  }
  ```
