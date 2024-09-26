# FairML.jl

## Citing



FairML.jl is a package developed for fair predictions. The package operates under a three-step framework:

1. Preprocessing: This stage encompasses the implementation of functions that perform initial data manipulation aimed at enhancing fairness metrics.
2. In-Processing: This stage constitutes the main part of the paper, where optimization problems are addressed with the aim of improving a specific fairness metric.
3. Post-processing: Following the previous stage, which outputs class membership probabilities, this phase is responsible for performing classification. It may or may not employ strategies to optimize a specific fairness metric in relation to accuracy.




The package's core functionality is a function that unifies all stages into a single, user-friendly interface.

If you use this project in your work, please cite it as follows:
```bibtex
@article{Author2024,
  author = {First Last et al.},
  title = {Project Title},
  journal = {Journal Name},
  year = {2024},
  volume = {15},
  number = {2},
  pages = {100-110},
  doi = {10.1007/xxxxxx}
}
```
