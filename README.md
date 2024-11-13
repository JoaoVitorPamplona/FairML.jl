<a href="https://github.com/JoaoVitorPamplona/FairML.jl">
  <img width="150" align="left" src="https://github.com/JoaoVitorPamplona/FairML.jl/blob/main/FairML.png">
</a>

# Welcome to FairML (Fair Machine Learning)

FairML.jl is a Julia package designed for fair machine learning, focusing on both regular and mixed models. It employs optimization techniques to enhance fairness metrics, including disparate impact and disparate mistreatment.


The package operates under a three-step framework:
1. Preprocessing: This stage encompasses the implementation of functions that perform initial data manipulation aimed at enhancing fairness metrics.
2. In-Processing: This stage constitutes the main part of the package, where optimization problems are addressed with the aim of improving a specific fairness metric.
3. Post-processing: Following the previous stage, which outputs class membership probabilities, this phase is responsible for performing classification. It may or may not employ strategies to optimize a specific fairness metric in relation to accuracy.

The package's core functionality is a function that unifies all stages into a single, user-friendly interface. For the regular models we have:
 
```julia
function fair_pred(xtrain::DataFrame, ytrain::Vector{Union{Float64, Int64}}, newdata::DataFrame,
                  inprocess::Function, SF::Array{String}, preprocess::Function=id_pre,
                  postprocess::Function=id_post, c::Real=0.1, R::Int64=1, seed::Int64=42,
                  SFpre::String, SFpost::String)
  return predictions
end
```

And for mixed models we have:
```julia
function me_fair_pred(xtrain::DataFrame, ytrain::Vector{Union{Float64, Int64}}, newdata::DataFrame,
                      group_id_train::CategoricalVector, group_id_newdata::CategoricalVector,      
                      inprocess::Function, SF::Array{String}, postprocess::Function=id_post,
                      c::Real=0.1, SFpost::String)
  return predictions
end
```

The complete package documentation can be found in the paper [FairML: A Julia package for Fair Machine Learning](https://arxiv.org/pdf/2405.06433)




## EXAMPLE

### For functions within the package:
```julia
xtrain,ytrain,newdata,ynewdata = create_data(100000, 0.01, [-2;0.4;0.8;0.5;3], "Logistic", 1, 42)
predictions = fair_pred(xtrain, ytrain, newdata, di_logreg, ["x4"], id_pre, di_post, 0.1, 1, 42)
ac, fpr, fnr, tpr, tnr, rc, TP, FP, TN, FN = final_metrics(ynewdata, predictions)
FM = disparate_impact_metric(newdata, predictions, ["x4"])
```


### For functions of the MLJ.jl package:
```julia
xtrain2,ytrain2,newdata2,ynewdata2 = create_data(100000, 0.01, [-2;0.4;0.8;0.5;3], "Linear", 1, 42)
predictions2 = fair_pred(xtrain2, ytrain2, newdata2, RandomForestClassifier(), ["x4"], di_pre, di_post, 0.1, 5, 42)
ac2, fpr2, fnr2, tpr2, tnr2, rc2, TP2, FP2, TN2, FN2 = final_metrics(ynewdata2, predictions2)
FM2 = disparate_impact_metric(newdata2, predictions2, ["x4"])
```


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
