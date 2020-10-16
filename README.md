# Controlled  Peptide Sequence Generation With Deep Autoencoder
# Setup 
+ setup own packages: create your own conda Env, or update existing
+ Use the amp_gen.yml to create your own conda env to run this project. 
+ Use this command `conda-env create -n myEnvName -f amp_gen.yml`

# Usage
+ `./run.sh`. This will run with default config from `cfg.py`. Since `cfg.runname=default` output goes to `output/default` and `tb/default`.
    - `python main.py --tiny 1` for fast testing with default config file. This will also have `cfg.runname=default` so output goes to `output/default` and `tb/default`.


# Citations

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
