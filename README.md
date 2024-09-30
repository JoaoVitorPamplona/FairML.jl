<a href="https://github.com/JoaoVitorPamplona/FairML.jl">
  <img width="150" align="left" src="https://github.com/JoaoVitorPamplona/FairML.jl/blob/main/FairML.png">
</a>

# Welcome to FairML (Fair Machine Learning) package

FairML.jl is a package developed for fair predictions, in regular and in mixed models. The package operates under a three-step framework:

1. Preprocessing: This stage encompasses the implementation of functions that perform initial data manipulation aimed at enhancing fairness metrics.
2. In-Processing: This stage constitutes the main part of the paper, where optimization problems are addressed with the aim of improving a specific fairness metric.
3. Post-processing: Following the previous stage, which outputs class membership probabilities, this phase is responsible for performing classification. It may or may not employ strategies to optimize a specific fairness metric in relation to accuracy.

The package's core functionality is a function that unifies all stages into a single, user-friendly interface. For the regular models we have:
 
```julia
function fair_pred(xtrain::DataFrame, ytrain::Vector{Union{Float64, Int64}}, newdata::DataFrame, inprocess::Function, SF::Array{String}, preprocess::Function=id_pre,
                   postprocess::Function=ID_Post, c::Real=0.1, R::Int64=1, seed::Int64=42, SFpre::String, SFpost::String)
  return predictions
end
```

And for the mixed models we have:
```julia
function me_fair_pred(xtrain::DataFrame, ytrain::Vector{Union{Float64, Int64}}, newdata::DataFrame, group_id_train::CategoricalVector, group_id_newdata::CategoricalVector,      
                      inprocess::Function, SF::Array{String}, postprocess::Function=ID_Post, c::Real=0.1, SFpost::String)
  return predictions
end
```

The complete package documentation can be found in the paper [FairML: A Julia package for Fair Machine Learning](https://arxiv.org/pdf/2405.06433)



## Citing

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
