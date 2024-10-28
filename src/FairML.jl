module FairML

using SparseArrays, JLD, MLJ, BenchmarkTools, EvalMetrics, CategoricalArrays, MixedModels, Random, JuMP, Ipopt, RDatasets, LinearAlgebra, Statistics, DataFrames, DelimitedFiles, CSVFiles, CSV, StatsBase, FreqTables, Missings, LazyStack, DataStructures, Distributions, GLM, NearestNeighbors, BetaML, NearestNeighborModels

export create_data
export id_pre
export di_pre
export id_logreg
export di_logreg
export fnr_logreg
export fpr_logreg
export dm_logreg
export id_me_logreg
export di_me_logreg
export fnr_me_logreg
export fpr_me_logreg
export dm_me_logreg
export id_svm
export di_svm
export fnr_svm
export fpr_svm
export dm_svm
export id_me_svm
export di_me_svm
export fnr_me_svm
export fpr_me_svm
export dm_me_svm
export id_post
export di_post
export fnr_post
export fpr_post
export tnr_post
export tpr_post
export dm_post
export final_metrics
export disparate_impact_metric
export false_negative_rate_metric
export false_positive_rate_metric
export true_negative_rate_metric
export true_positive_rate_metric
export disparate_mistreatment_metric
export fair_pred
export me_fair_pred




"""
### Data generator
A synthetic dataset generation functions if the user wish to generate any for their own use.


    (xtrain, ytrain, newdata, ynewdata) = create_data(num_points::Int64, split::Float64, 
                                                      predictors::Array{Float64}, model::String, 
                                                      nSF::Int64, seed::Int64)  


#### Input arguments

* `num_points`: a linear operator that models a Hermitian positive definite matrix of dimension n;
* `split`: Proportion of the dataset will be utilized as the training set. The complementary proportion will generate the test dataset;
* `predictors`: The number of predictors in the model will correspond directly to the number of variables included in your dataset;
* `model`: Linear or Logistic;
* `nSF`: Number of sensitive features in the dataset;
* `seed`: Random seed.


#### Output arguments

* `xtrain`: The dataset that the labels are known;
* `ytrain`: The labels of the dataset `xtrain`;
* `newdata`: A new dataset used for testing;
* `ynewdata`: The labels of the dataset `newdata`.
"""

function create_data(num_points::Int64, split::Float64, predictors::Union{Vector{Int64},Vector{Float64}}, model::String, nSF::Int64, seed::Int64)
    SF = rand(MersenneTwister(seed), num_points, nSF) .< 0.5
    mean = zeros(length(predictors)-(nSF+1))
    C = Float64.(1*Matrix(I, length(predictors)-(nSF+1), length(predictors)-(nSF+1)))
    d = MvNormal(mean, C)
    X = [ones(num_points) rand(MersenneTwister(seed), d, num_points)' SF]

    if model == "Logistic"
        m12(x,predictors) = 1/(1 + exp(- dot(predictors,x)))
        ly = size(X)[1]
        verificar = zeros(ly)
        y = zeros(ly)
    
        d = [Binomial(1,m12(X[i,:],predictors)) for i=1:ly]
        y = [rand(MersenneTwister(i),d[i], 1)[1] for i=1:ly]
        target = [y == 1 ? 1 : -1 for y in y]
    elseif model == "Linear"
        m13(x,predictors) = dot(predictors,x)
        ly = size(X)[1]
        verificar = zeros(ly)
        y = zeros(ly)
        for i = 1 : ly
            a = m13(X[i,:],predictors)
            verificar[i] = exp(a)/(1+exp(a))
            d = [Binomial(1, verificar[i])]
            y[i] = rand(MersenneTwister(42),d[1], 1)[1]        
        end
        target = [y == 1 ? 1 : -1 for y in y]
    end

    CDT = [X[:,2:end] target]
    CDT = DataFrame(CDT, :auto)

    CDT = shuffle(MersenneTwister(42), CDT)
    df1 = CDT[1:Int64(round(split*size(CDT,1))),:]
    df2 = CDT[Int64(round(split*size(CDT,1)))+1:end,:] 

    xtrain = df1[:,1:end-1]
    ytrain = df1[:,end]
    newdata = df2[:,1:end-1]
    ynewdata = df2[:,end]


    return xtrain,ytrain,newdata,ynewdata
end




"""
###Prediction functions 
The package's core functionality, a functions that unifies preprocessing, in-processing and post-processing phases into a single, user-friendly interface.

### For regular models...
    predictions = fair_pred(xtrain::DataFrame, ytrain::Vector{Union{Float64, Int64}}, newdata::DataFrame, inprocess::Function, 
                            SF::Array{String}, preprocess::Function=id_pre, postprocess::Function=id_post, c::Real=0.1, 
                            R::Int64=1, seed::Int64=42, SFpre::String="0", SFpost::String="0")

### and for mixed models
    predictions = me_fair_pred(xtrain::DataFrame, ytrain::Vector{Union{Float64, Int64}}, newdata::DataFrame, group_id_train::CategoricalVector, 
                               group_id_newdata::CategoricalVector, inprocess::Function, SF::Array{String}, postprocess::Function=id_post, 
                               c::Real=0.1, SFpost::String="0")




#### Input arguments

* `xtrain`: The dataset that the labels are known (training set);
* `ytrain`: The labels of the dataset `xtrain`;
* `newdata`: The new dataset for which we want to obtain the `predictions`;
* `inprocess`: One of the several optimization problems availables in this package or any machine learning classification method present in MLJ.jl package;
* `SF`: One or a set of sensitive features (variables names), that will act in the in-processing phase. 
        If the algorithm come from the MLJ.jl package, no fair constraint are acting in this phase;
* `group_id_train`: Training set group category;
* `group_id_newdata`: New dataset group category;


#### Optional argument

* `preprocess`: A pre-processing function among the options available in this package, `id_pre()` by default;
* `postprocess`: A post-processing function among the options available in this package, `id_post()` by default;
* `c`: The threshold of the fair optimization problems, 0.1 by default;
* `R`: Number of iterations of the preprocessing phase, each time sampling differently using the resampling method, 1 by default;
* `seed`: For sample selection in `R`, 42 by default;
* `SFpre`: One sensitive features (variable name), that will act in the preprocessing phase, disabled by default;
* `SFpost`: One sensitive features (variable name), that will act in the post-processing phase, disabled by default.


#### Output arguments

* `predictions`: Classification of the `newdata` points.
"""
function fair_pred(xtrain::DataFrame, ytrain::Vector, newdata::DataFrame, inprocess, SF::Array{String}, preprocess::Function=id_pre, postprocess::Function=id_post, c::Real=0.1,  R::Int64=1, seed::Int64=42, SFpre::String="0", SFpost::String="0")
    if all(x -> x == 0 || x == 1, Matrix(xtrain[!,SF])) == false && all(x -> x == 0.0 || x == 1.0, Matrix(xtrain[!,SF])) == false
        error("SF must have just 1 or 0 values.")
    end
    if all(x -> x == 0 || x == 1 || x == 1.0 || x == 0.0, ytrain) == false && all(x -> x == -1 || x == 1 || x == -1.0 || x == 1.0, ytrain) == false
        error("ytrain must have just 1 and 0 values or 1 and -1 values.")
    end
    if SFpost == "0"
        SFpost = SF
    end
    if SFpre == "0"
        SFpre = SF
    end
    if all(x -> x == 0 || x == 1, Matrix(xtrain[!,SFpost])) == false && all(x -> x == 0.0 || x == 1.0, Matrix(xtrain[!,SFpost])) == false
        error("SFpost must have just 1 or 0 values.")
    end
    if all(x -> x == 0 || x == 1, Matrix(xtrain[!,SFpre])) == false && all(x -> x == 0.0 || x == 1.0, Matrix(xtrain[!,SFpre])) == false
        error("SFpre must have just 1 or 0 values.")
    end
    c = abs(c)
    if R == 1
        xtrain, ytrain, newdata = preprocess(xtrain, ytrain, newdata, SFpre, c, seed)
        if isa(inprocess, Function) == false
            ytrain = categorical([y == 1 ? 1 : 0 for y in ytrain])
            model = inprocess
            mach = machine(model, xtrain, ytrain)
            MLJ.fit!(mach)
            prob_train = MLJ.predict(mach, xtrain)
            prob_newdata = MLJ.predict(mach, newdata)
            if typeof(prob_train[1]) == UnivariateFinite{Multiclass{2}, Int64, UInt32, Float64}
                prob_train =  broadcast(pdf, prob_train, 1)
                prob_newdata =  broadcast(pdf, prob_newdata, 1)
            else
                prob_train = levelcode.(prob_train)
                prob_newdata = levelcode.(prob_newdata)
                prob_train = [y == 1 ? 1 : 0 for y in prob_train]
                prob_newdata = [y == 1 ? 1 : 0 for y in prob_newdata]
            end
        else
            prob_train,prob_newdata = inprocess(xtrain, ytrain, newdata, SF, c)
        end
        predictions = postprocess(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost)
    else
        DI_Metric1_old = 2
        sed = abs.(rand(MersenneTwister(seed), Int, R))
        for l = 1 : R
            xtrain3, ytrain3, newdata3 = preprocess(xtrain, ytrain, newdata, SFpre, c, sed[l]) 
            if isa(inprocess, Function) == false
                ytrain3 = categorical([y == 1 ? 1 : 0 for y in ytrain3])
                model = inprocess
                mach = machine(model, xtrain3, ytrain3)
                MLJ.fit!(mach)
                prob_train3 = MLJ.predict(mach, xtrain3)
                prob_train = MLJ.predict(mach, xtrain)
                if typeof(prob_train3[1]) == UnivariateFinite{Multiclass{2}, Int64, UInt32, Float64}
                    prob_train3 =  broadcast(pdf, prob_train3, 1)
                    prob_train =  broadcast(pdf, prob_train, 1)
                else
                    prob_train3 = levelcode.(prob_train3)
                    prob_train = levelcode.(prob_train)
                    prob_train3 = [y == 1 ? 1 : 0 for y in prob_train3]
                    prob_train = [y == 1 ? 1 : 0 for y in prob_train]
                end
            else
                prob_train3, prob_train = inprocess(xtrain3, ytrain3, xtrain, SF, c)
            end
            predictions_Train = postprocess(prob_train3,prob_train,xtrain3,ytrain3,xtrain,SFpost)
            #if isa(inprocess, Function) == false && any(xtrain3[:, 1] .== 1.0) && any(xtrain3[:, end] .== 1.0)
            #    select!(xtrain3,Not([:Intercept]))
            #end
            DI_Metric1 = disparate_impact_metric(xtrain, predictions_Train, SFpre)
            #if isa(inprocess, Function) == false && any(xtrain[:, 1] .== 1.0) && any(xtrain[:, end] .== 1.0)
            #    select!(xtrain,Not([:Intercept]))
            #end
            if DI_Metric1 < DI_Metric1_old
                DI_Metric1_old = copy(DI_Metric1)
                if isa(inprocess, Function) == false
                    ytrain3 = categorical([y == 1 ? 1 : 0 for y in ytrain3])
                    model = inprocess
                    mach = machine(model, xtrain3, ytrain3)
                    MLJ.fit!(mach)
                    prob_train = MLJ.predict(mach, xtrain3)
                    prob_newdata = MLJ.predict(mach, newdata3)
                    if typeof(prob_train[1]) == UnivariateFinite{Multiclass{2}, Int64, UInt32, Float64}
                        prob_train =  broadcast(pdf, prob_train, 1)
                        prob_newdata =  broadcast(pdf, prob_newdata, 1)
                    else
                        prob_train = levelcode.(prob_train)
                        prob_newdata = levelcode.(prob_newdata)
                        prob_train = [y == 1 ? 1 : 0 for y in prob_train]
                        prob_newdata = [y == 1 ? 1 : 0 for y in prob_newdata]
                    end
                else
                    prob_train,prob_newdata = inprocess(xtrain3, ytrain3, newdata3, SF, c)
                end
                predictions = postprocess(prob_train,prob_newdata,xtrain3,ytrain3,newdata3,SFpost)
            end 
        end
    end
    return predictions 
end


function me_fair_pred(xtrain::DataFrame, ytrain::Vector, newdata::DataFrame, group_id_train::CategoricalVector, group_id_newdata::CategoricalVector, inprocess::Function, SF::Array{String}, postprocess::Function=id_post, c::Real=0.1, SFpost::String="0")
    if all(x -> x == 0 || x == 1, Matrix(xtrain[!,SF])) == false && all(x -> x == 0.0 || x == 1.0, Matrix(xtrain[!,SF])) == false
        error("SF must have just 1 or 0 values.")
    end
    if all(x -> x == 0 || x == 1 || x == 1.0 || x == 0.0, ytrain) == false && all(x -> x == -1 || x == 1 || x == -1.0 || x == 1.0, ytrain) == false
        error("ytrain must have just 1 and 0 values or 1 and -1 values.")
    end
    if SFpost == "0"
        SFpost = SF
    end
    if all(x -> x == 0 || x == 1, Matrix(xtrain[!,SFpost])) == false && all(x -> x == 0.0 || x == 1.0, Matrix(xtrain[!,SFpost])) == false
        error("SFpost must have just 1 or 0 values.")
    end
    c = abs(c)
    prob_train, prob_newdata  = inprocess(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    predictions = postprocess(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost)
    return predictions 
end





"""
###Preprocessing phase
Functions that preprocess the data. The id_pre() function does not modify the data in any way and is used by default. The di_pre() function forces the data to be free of disparate impact.

    xtrain, ytrain, newdata = preprocess(xtrain::DataFrame, ytrain::Vector{Union{Float64, Int64}}, newdata::DataFrame, SFpre::String, c::Real, seed::Int64)



#### Input arguments

* `xtrain`: The dataset that the labels are known (training set);
* `ytrain`: The labels of the dataset `xtrain`;
* `newdata`: The new dataset for which we want to obtain the `predictions`;
* `SFpre`: One sensitive features (variable name), that will act in the preprocessing phase, disabled by default;
* `c`: The threshold of the fair optimization problems, 0.1 by default;
* `seed`: For sample selection, 42 by default;


#### Optional argument

* `preprocess`: A pre-processing function among the options available in this package, `id_pre()` by default;
* `postprocess`: A post-processing function among the options available in this package, `id_post()` by default;
* `c`: The threshold of the fair optimization problems, 0.1 by default;
* `R`: Number of iterations of the preprocessing phase, each time sampling differently using the resampling method, 1 by default;
* `seed`: For sample selection in `R`, 42 by default;
* `SFpre`: One sensitive features (variable name), that will act in the preprocessing phase, disabled by default;
* `SFpost`: One sensitive features (variable name), that will act in the post-processing phase, disabled by default.


#### Output arguments

* `xtrain`: New training set, after the resampling method;
* `ytrain`: New training set labels, after the resampling method;
* `newdata`: It remains unchanged during the pre-processing phase;
"""
function id_pre(xtrain, ytrain, newdata, SFpre, c, seed)
    return xtrain, ytrain, newdata
end


function di_pre(xtrain, ytrain, newdata, SFpre, c, seed)
    if c == 0
        c = 0.01
    end
    Data = hcat(xtrain, ytrain)

    gb = groupby(Data, SFpre)
    c1 = countmap(gb[1].x1)
    c2 = countmap(gb[2].x1)
    mm = minimum(vcat(collect(values(c1)), collect(values(c2))))

    gb11 = groupby(gb[1], :x1)
    gb12 = groupby(gb[2], :x1)

    gb1 = gb11[1]
    gb2 = gb11[2]
    gb3 = gb12[1]
    gb4 = gb12[2]

    selected_indices1 = sample(MersenneTwister(seed), 1:nrow(gb1), mm, replace=true)
    selected_indices2 = sample(MersenneTwister(seed), 1:nrow(gb2), mm, replace=true)
    selected_indices3 = sample(MersenneTwister(seed), 1:nrow(gb3), mm, replace=true)
    selected_indices4 = sample(MersenneTwister(seed), 1:nrow(gb4), mm, replace=true)

    df_sampled1 = gb1[selected_indices1, :]
    df_sampled2 = gb2[selected_indices2, :]
    df_sampled3 = gb3[selected_indices3, :]
    df_sampled4 = gb4[selected_indices4, :]


    ND = shuffle(MersenneTwister(seed), vcat(df_sampled1, df_sampled2, df_sampled3, df_sampled4))
    xtrainF = ND[:,1:end-1]
    ytrainF = ND[:,end]
    return xtrainF, ytrainF, newdata
end




#In-processing phase
function id_logreg(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @NLexpression(model, hi[i=1:m], 1/(1 + exp(- sum(β[j] * xtrain[i,j] for j=1:n))))

    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(hi[i]) + ((1 - ytrain[i])/2) * log(1 - hi[i]) for i = 1:m ))

    optimize!(model)
    
    Pred = JuMP.value.(β)

    m1(x,β) = 1/(1 + exp(- dot(β,x)))
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        prob_newdata[i] = m1(newdata[i,:],Pred)
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        prob_train[i] = m1(xtrain[i,:],Pred)
    end

    return prob_train, prob_newdata
end

function di_logreg(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    m,n = size(xtrain)
    try 
        global Z = Vector(xtrain[!,SF])
    catch
        global Z = Matrix(xtrain[!,SF])
    end
    zeh = Z .- Statistics.mean(Z, dims=1)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @NLexpression(model, hi[i=1:m], 1/(1 + exp(- sum(β[j] * xtrain[i,j] for j=1:n))))

    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(hi[i]) + ((1 - ytrain[i])/2) * log(1 - hi[i]) for i = 1:m ))

    @constraint(model, [i=1:size(Z,2)], (1/m)*sum(zeh[p,i]*sum(β[j]*xtrain[p,j] for j=1:n) for p = 1:m) ≤ c)
    @constraint(model, [i=1:size(Z,2)], (1/m)*sum(zeh[p,i]*sum(β[j]*xtrain[p,j] for j=1:n) for p = 1:m) ≥ -c)

    optimize!(model)
    
    Pred = JuMP.value.(β)
    m1(x,β) = 1/(1 + exp(- dot(β,x)))
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        prob_newdata[i] = m1(newdata[i,:],Pred)
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        prob_train[i] = m1(xtrain[i,:],Pred)
    end

    return prob_train, prob_newdata
end

function fnr_logreg(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    m,n = size(xtrain)
    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    N02 = zeros(Int64,length(SF))
    N12 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        N02[i] = size(a0,1)
        N12[i] = size(a1,1)
        fa0 = findall(==(1), a0[:,end])
        fa1 = findall(==(1), a1[:,end])
        a0 = a0[fa0,:]
        a1 = a1[fa1,:]
        xtrain0[i] = a0[:,1:end-1]
        xtrain1[i] = a1[:,1:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]]
    end



    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @NLexpression(model, hi[i=1:m], 1/(1 + exp(- sum(β[j] * xtrain[i,j] for j=1:n))))

    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(hi[i]) + ((1 - ytrain[i])/2) * log(1 - hi[i]) for i = 1:m ))

    @NLconstraint(model, [i=1:length(SF)], (-N12[i]/m)*sum(min(0, sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N02[i]/m)*sum(min(0, sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≤ c)
    @NLconstraint(model, [i=1:length(SF)], (-N12[i]/m)*sum(min(0, sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N02[i]/m)*sum(min(0, sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≥ -c)

    optimize!(model)
    
    Pred = JuMP.value.(β)
    m1(x,β) = 1/(1 + exp(- dot(β,x)))
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        prob_newdata[i] = m1(newdata[i,:],Pred)
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        prob_train[i] = m1(xtrain[i,:],Pred)
    end

    return prob_train, prob_newdata
end

function fpr_logreg(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    m,n = size(xtrain)
    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    N02 = zeros(Int64,length(SF))
    N12 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        N02[i] = size(a0,1)
        N12[i] = size(a1,1)
        fa0 = findall(==(-1), a0[:,end])
        fa1 = findall(==(-1), a1[:,end])
        a0 = a0[fa0,:]
        a1 = a1[fa1,:]
        xtrain0[i] = a0[:,1:end-1]
        xtrain1[i] = a1[:,1:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]]
    end



    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])

    @NLexpression(model, hi[i=1:m], 1/(1 + exp(- sum(β[j] * xtrain[i,j] for j=1:n))))
    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(hi[i]) + ((1 - ytrain[i])/2) * log(1 - hi[i]) for i = 1:m ))

    @NLconstraint(model, [i=1:length(SF)], (-N12[i]/m)*sum(min(0, -sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N02[i]/m)*sum(min(0, -sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≤ c)
    @NLconstraint(model, [i=1:length(SF)], (-N12[i]/m)*sum(min(0, -sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N02[i]/m)*sum(min(0, -sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≥ -c)

    optimize!(model)
    
    Pred = JuMP.value.(β)
    m1(x,β) = 1/(1 + exp(- dot(β,x)))
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        prob_newdata[i] = m1(newdata[i,:],Pred)
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        prob_train[i] = m1(xtrain[i,:],Pred)
    end

    return prob_train, prob_newdata
end

function dm_logreg(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    m,n = size(xtrain)
    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        xtrain0[i] = a0[:,1:end-1]
        xtrain1[i] = a1[:,1:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]]
    end



    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @NLexpression(model, hi[i=1:m], 1/(1 + exp(- sum(β[j] * xtrain[i,j] for j=1:n))))

    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(hi[i]) + ((1 - ytrain[i])/2) * log(1 - hi[i]) for i = 1:m ))

    @NLconstraint(model, [i=1:length(SF)], (-N1[i]/m)*sum(min(0, ((1-ytrain0[i][k])/2)*ytrain0[i][k]*sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N0[i]/m)*sum(min(0, ((1-ytrain1[i][k])/2)*ytrain1[i][k]*sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≤ c)
    @NLconstraint(model, [i=1:length(SF)], (-N1[i]/m)*sum(min(0, ((1-ytrain0[i][k])/2)*ytrain0[i][k]*sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N0[i]/m)*sum(min(0, ((1-ytrain1[i][k])/2)*ytrain1[i][k]*sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≥ -c)

    @NLconstraint(model, [i=1:length(SF)], (-N1[i]/m)*sum(min(0, ((1+ytrain0[i][k])/2)*ytrain0[i][k]*sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N0[i]/m)*sum(min(0, ((1+ytrain1[i][k])/2)*ytrain1[i][k]*sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≤ c)
    @NLconstraint(model, [i=1:length(SF)], (-N1[i]/m)*sum(min(0, ((1+ytrain0[i][k])/2)*ytrain0[i][k]*sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N0[i]/m)*sum(min(0, ((1+ytrain1[i][k])/2)*ytrain1[i][k]*sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≥ -c)

    optimize!(model)
    
    Pred = JuMP.value.(β)
    m1(x,β) = 1/(1 + exp(- dot(β,x)))
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        prob_newdata[i] = m1(newdata[i,:],Pred)
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        prob_train[i] = m1(xtrain[i,:],Pred)
    end

    return prob_train, prob_newdata
end


function id_me_logreg(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain]
    sorted_indices = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices, :]

    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    m,n = size(xtrain)


    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
  
    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) for i = 1 : nct[1]) - sum(sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) for i = sum(nct[r] for r = 1 : l-1)+1 : sum(nct[r] for r = 1 : l-1) + nct[l]) for l = 2 : nc) + sum(b[i]^2 for i = 1 : nc))

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = 1/(1 + exp(- dot(PredFix,x) - PredRand))
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end

function di_me_logreg(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain]
    sorted_indices = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices, :]

    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    m,n = size(xtrain)
    
    try 
        global Z = Vector(xtrain[!,SF])
    catch
        global Z = Matrix(xtrain[!,SF])
    end
    means_z = Statistics.mean(Z, dims=1)

    zeh = zeros(m,length(SF))
    for p = 1 : length(SF)
        zeh[1:nct[1],p] = Z[1:nct[1],p] .- means_z[1,p]
    end
    for p = 1 : length(SF)
        for i = 2 : nc
            zeh[sum(nct[j] for j = 1 : i-1)+1:sum(nct[j] for j = 1 : i-1)+nct[i],p] = Z[sum(nct[j] for j = 1 : i-1)+1:sum(nct[j] for j = 1 : i-1)+nct[i],p] .- means_z[1,p]
        end
    end

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
  
    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) for i = 1 : nct[1]) - sum(sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) for i = sum(nct[r] for r = 1 : l-1)+1 : sum(nct[r] for r = 1 : l-1) + nct[l]) for l = 2 : nc) + sum(b[i]^2 for i = 1 : nc))

    
    @constraint(model, [k=1:size(Z,2)], ((1/m)*sum(zeh[j,k]*(dot(β, xtrain[j,:]) + b[1]) for j = 1 : nct[1]) + (1/m)*sum(sum(zeh[j,k]*(dot(β, xtrain[j,:]) + b[i]) for j = sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]) for i = 2 : nc)) ≤ c)
    @constraint(model, [k=1:size(Z,2)], ((1/m)*sum(zeh[j,k]*(dot(β, xtrain[j,:]) + b[1]) for j = 1 : nct[1]) + (1/m)*sum(sum(zeh[j,k]*(dot(β, xtrain[j,:]) + b[i]) for j = sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]) for i = 2 : nc)) ≥ -c)

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = 1/(1 + exp(- dot(PredFix,x) - PredRand))
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end

function fnr_me_logreg(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    xtrain = [group_id_train xtrain]
    xtrain = rename(xtrain,:x1 => :ClusterID)
    xtrain = sort(xtrain, :ClusterID)
    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    

    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    nct0 = zeros(Int64, nc, length(SF))
    nct1 = zeros(Int64, nc, length(SF))
    N02 = zeros(Int64,length(SF))
    N12 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        N02[i] = size(a0,1)
        N12[i] = size(a1,1)
        fa0 = findall(==(1), a0[:,end])
        fa1 = findall(==(1), a1[:,end])
        a0 = a0[fa0,:]
        a1 = a1[fa1,:]
        for num in a0.ClusterID
            nct0[num,i] += 1
        end
        for num in a1.ClusterID
            nct1[num,i] += 1
        end
        xtrain0[i] = a0[:,2:end-1]
        xtrain1[i] = a1[:,2:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]] 
    end
    xtrain = xtrain[:,2:end]
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
  
    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) for i = 1 : nct[1]) - sum(sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) for i = sum(nct[r] for r = 1 : l-1)+1 : sum(nct[r] for r = 1 : l-1) + nct[l]) for l = 2 : nc) + sum(b[i]^2 for i = 1 : nc))

    @NLconstraint(model, [k=1:length(SF)], ((-N1[k]/m)*(sum(min(0,(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[1])) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,((sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[i]))) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0[k]/m)*(sum(min(0,(sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[1])) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[i])) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) ≤ c)   
    @NLconstraint(model, [k=1:length(SF)], ((-N1[k]/m)*(sum(min(0,(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[1])) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,((sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[i]))) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0[k]/m)*(sum(min(0,(sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[1])) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[i])) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) ≥ -c)   

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = 1/(1 + exp(- dot(PredFix,x) - PredRand))
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end

function fpr_me_logreg(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    xtrain = [group_id_train xtrain]
    xtrain = rename(xtrain,:x1 => :ClusterID)
    xtrain = sort(xtrain, :ClusterID)
    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    

    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    nct0 = zeros(Int64, nc, length(SF))
    nct1 = zeros(Int64, nc, length(SF))
    N02 = zeros(Int64,length(SF))
    N12 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        N02[i] = size(a0,1)
        N12[i] = size(a1,1)
        fa0 = findall(==(-1), a0[:,end])
        fa1 = findall(==(-1), a1[:,end])
        a0 = a0[fa0,:]
        a1 = a1[fa1,:]
        for num in a0.ClusterID
            nct0[num,i] += 1
        end
        for num in a1.ClusterID
            nct1[num,i] += 1
        end
        xtrain0[i] = a0[:,2:end-1]
        xtrain1[i] = a1[:,2:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]] 
    end
    xtrain = xtrain[:,2:end]
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
  
    @NLobjective(model, Min, -sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) for i = 1 : nct[1]) - sum(sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) for i = sum(nct[r] for r = 1 : l-1)+1 : sum(nct[r] for r = 1 : l-1) + nct[l]) for l = 2 : nc) + sum(b[i]^2 for i = 1 : nc))
   
    @NLconstraint(model, [k=1:length(SF)], ((-N1[k]/m)*(sum(min(0,-(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[1])) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(-(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[i]))) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0[k]/m)*(sum(min(0,(-sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[1])) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(-sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[i])) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) ≤ c)   
    @NLconstraint(model, [k=1:length(SF)], ((-N1[k]/m)*(sum(min(0,-(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[1])) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(-(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[i]))) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0[k]/m)*(sum(min(0,(-sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[1])) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(-sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[i])) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) ≥ -c)   

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = 1/(1 + exp(- dot(PredFix,x) - PredRand))
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end

function dm_me_logreg(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    xtrain = [group_id_train xtrain]
    xtrain = rename(xtrain,:x1 => :ClusterID)
    xtrain = sort(xtrain, :ClusterID)
    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    
    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    nct0 = zeros(Int64, nc, length(SF))
    nct1 = zeros(Int64, nc, length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        for num in a0.ClusterID
            nct0[num,i] += 1
        end
        for num in a1.ClusterID
            nct1[num,i] += 1
        end
        xtrain0[i] = a0[:,2:end-1]
        xtrain1[i] = a1[:,2:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]]  
    end
    xtrain = xtrain[:,2:end]
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
  
    @objective(model, Min, -sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n)+ b[1]) ))) for i = 1 : nct[1]) - sum(sum(((ytrain[i]+1)/2) * log(1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) + ((1 - ytrain[i])/2) * log(1 - 1/(1 + exp(- (sum(β[k] * xtrain[i,k] for k = 1 : n) + b[l]) ))) for i = sum(nct[r] for r = 1 : l-1)+1 : sum(nct[r] for r = 1 : l-1) + nct[l]) for l = 2 : nc) + sum(b[i]^2 for i = 1 : nc))
 
    @constraint(model, [k=1:length(SF)], ((-N1/m)*(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[1]) * ((1+ytrain0[k][j])/2)*ytrain0[k][j]) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[i]) * ((1+ytrain0[k][j])/2)*ytrain0[k][j]) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0/m)*(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[1]) * ((1+ytrain1[k][j])/2)*ytrain1[k][j]) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[i]) * ((1+ytrain1[k][j])/2)*ytrain1[k][j]) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) .≤ c)   
    @constraint(model, [k=1:length(SF)], ((-N1/m)*(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[1]) * ((1+ytrain0[k][j])/2)*ytrain0[k][j]) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[i]) * ((1+ytrain0[k][j])/2)*ytrain0[k][j]) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0/m)*(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[1]) * ((1+ytrain1[k][j])/2)*ytrain1[k][j]) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[i]) * ((1+ytrain1[k][j])/2)*ytrain1[k][j]) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) .≥ -c)

    @constraint(model, [k=1:length(SF)], ((-N1/m)*(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[1]) * ((1-ytrain0[k][j])/2)*ytrain0[k][j]) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[i]) * ((1-ytrain0[k][j])/2)*ytrain0[k][j]) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0/m)*(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[1]) * ((1-ytrain1[k][j])/2)*ytrain1[k][j]) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[i]) * ((1-ytrain1[k][j])/2)*ytrain1[k][j]) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) .≤ c)   
    @constraint(model, [k=1:length(SF)], ((-N1/m)*(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[1]) * ((1-ytrain0[k][j])/2)*ytrain0[k][j]) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[i]) * ((1-ytrain0[k][j])/2)*ytrain0[k][j]) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0/m)*(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[1]) * ((1-ytrain1[k][j])/2)*ytrain1[k][j]) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[i]) * ((1-ytrain1[k][j])/2)*ytrain1[k][j]) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) .≥ -c)

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = 1/(1 + exp(- dot(PredFix,x) - PredRand))
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end




function id_svm(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    if sort(collect(keys(countmap(ytrain)))) != collect([-1, 1])
        error("ytrain must have just -1 or 1.")
    end
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ))
    @constraint(model, [i=1:m], (dot(β, xtrain[i,:])) * ytrain[i] ≥ 1 - ξ[i])

    optimize!(model)
    Pred = JuMP.value.(β)
    
    m1(x,β) = dot(β,x)
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        a = m1(newdata[i,:],Pred)
        prob_newdata[i] = exp(a)/(1+exp(a))
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        a = m1(xtrain[i,:],Pred)
        prob_train[i] = exp(a)/(1+exp(a))
    end

    return prob_train,prob_newdata
end

function di_svm(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain]
    m,n = size(xtrain)
    try 
        global Z = Vector(xtrain[!,SF])
    catch
        global Z = Matrix(xtrain[!,SF])
    end
    zeh = Z .- Statistics.mean(Z, dims=1)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ))
    @constraint(model, [i=1:m], (dot(β, xtrain[i,:])) * ytrain[i] ≥ 1 - ξ[i])

    @constraint(model, [i=1:size(Z,2)], (1/m)*sum(zeh[p,i]*sum(β[j]*xtrain[p,j] for j=1:n) for p = 1:m) ≤ c)
    @constraint(model, [i=1:size(Z,2)], (1/m)*sum(zeh[p,i]*sum(β[j]*xtrain[p,j] for j=1:n) for p = 1:m) ≥ -c)

    optimize!(model)
    
    Pred = JuMP.value.(β)
    m1(x,β) = dot(β,x)
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        a = m1(newdata[i,:],Pred)
        prob_newdata[i] = exp(a)/(1+exp(a))
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        a = m1(xtrain[i,:],Pred)
        prob_train[i] = exp(a)/(1+exp(a))
    end

    return prob_train,prob_newdata
end

function fnr_svm(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    m,n = size(xtrain)
    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    N02 = zeros(Int64,length(SF))
    N12 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        N02[i] = size(a0,1)
        N12[i] = size(a1,1)
        fa0 = findall(==(1), a0[:,end])
        fa1 = findall(==(1), a1[:,end])
        a0 = a0[fa0,:]
        a1 = a1[fa1,:]
        xtrain0[i] = a0[:,1:end-1]
        xtrain1[i] = a1[:,1:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]]
    end



    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ))
    @constraint(model, [i=1:m], (dot(β, xtrain[i,:])) * ytrain[i] ≥ 1 - ξ[i])

    @NLconstraint(model, [i=1:length(SF)], (-N12[i]/m)*sum(min(0, sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N02[i]/m)*sum(min(0, sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≤ c)
    @NLconstraint(model, [i=1:length(SF)], (-N12[i]/m)*sum(min(0, sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N02[i]/m)*sum(min(0, sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≥ -c)

    optimize!(model)
    
    Pred = JuMP.value.(β)
    m1(x,β) = dot(β,x)
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        a = m1(newdata[i,:],Pred)
        prob_newdata[i] = exp(a)/(1+exp(a))
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        a = m1(xtrain[i,:],Pred)
        prob_train[i] = exp(a)/(1+exp(a))
    end

    return prob_train,prob_newdata
end

function fpr_svm(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    m,n = size(xtrain)
    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    N02 = zeros(Int64,length(SF))
    N12 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        N02[i] = size(a0,1)
        N12[i] = size(a1,1)
        fa0 = findall(==(-1), a0[:,end])
        fa1 = findall(==(-1), a1[:,end])
        a0 = a0[fa0,:]
        a1 = a1[fa1,:]
        xtrain0[i] = a0[:,1:end-1]
        xtrain1[i] = a1[:,1:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]]
    end



    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ))
    @constraint(model, [i=1:m], (dot(β, xtrain[i,:])) * ytrain[i] ≥ 1 - ξ[i])

    @NLconstraint(model, [i=1:length(SF)], (-N12[i]/m)*sum(min(0, -sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N02[i]/m)*sum(min(0, -sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≤ c)
    @NLconstraint(model, [i=1:length(SF)], (-N12[i]/m)*sum(min(0, -sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N02[i]/m)*sum(min(0, -sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≥ -c)

    optimize!(model)
    
    Pred = JuMP.value.(β)
    m1(x,β) = dot(β,x)
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        a = m1(newdata[i,:],Pred)
        prob_newdata[i] = exp(a)/(1+exp(a))
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        a = m1(xtrain[i,:],Pred)
        prob_train[i] = exp(a)/(1+exp(a))
    end

    return prob_train,prob_newdata
end

function dm_svm(xtrain, ytrain, newdata, SF, c)
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    m,n = size(xtrain)
    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        xtrain0[i] = a0[:,1:end-1]
        xtrain1[i] = a1[:,1:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
    end



    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 300.0)
    @variable(model, β[1:n])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ))
    @constraint(model, [i=1:m], (dot(β, xtrain[i,:])) * ytrain[i] ≥ 1 - ξ[i])

    @NLconstraint(model, [i=1:length(SF)], (-N1[i]/m)*sum(min(0, ((1-ytrain0[i][k])/2)*ytrain0[i][k]*sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N0[i]/m)*sum(min(0, ((1-ytrain1[i][k])/2)*ytrain1[i][k]*sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≤ c)
    @NLconstraint(model, [i=1:length(SF)], (-N1[i]/m)*sum(min(0, ((1-ytrain0[i][k])/2)*ytrain0[i][k]*sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N0[i]/m)*sum(min(0, ((1-ytrain1[i][k])/2)*ytrain1[i][k]*sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≥ -c)

    @NLconstraint(model, [i=1:length(SF)], (-N1[i]/m)*sum(min(0, ((1+ytrain0[i][k])/2)*ytrain0[i][k]*sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N0[i]/m)*sum(min(0, ((1+ytrain1[i][k])/2)*ytrain1[i][k]*sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≤ c)
    @NLconstraint(model, [i=1:length(SF)], (-N1[i]/m)*sum(min(0, ((1+ytrain0[i][k])/2)*ytrain0[i][k]*sum(β[j] * xtrain0[i][k,j] for j=1:n)) for k = 1 : N0[i]) + (N0[i]/m)*sum(min(0, ((1+ytrain1[i][k])/2)*ytrain1[i][k]*sum(β[j] * xtrain1[i][k,j] for j=1:n)) for k = 1 : N1[i]) ≥ -c)

    optimize!(model)
    
    Pred = JuMP.value.(β)
    m1(x,β) = dot(β,x)
    prob_newdata = zeros(size(newdata,1))
    for i = 1 : size(newdata,1)
        a = m1(newdata[i,:],Pred)
        prob_newdata[i] = exp(a)/(1+exp(a))
    end

    prob_train = zeros(size(xtrain,1))
    for i = 1 : size(xtrain,1)
        a = m1(xtrain[i,:],Pred)
        prob_train[i] = exp(a)/(1+exp(a))
    end

    return prob_train,prob_newdata
end


function id_me_svm(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain]
    sorted_indices = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices, :]

    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ) + dot(b,b))
    
    @constraint(model, [j=1:nct[1]], (dot(β, xtrain[j,:]) + b[1]) * ytrain[j] ≥ 1 - ξ[j])
    for i = 2 : nc
        @constraint(model, [j=sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]], ((dot(β, xtrain[j,:]) + b[i]) * ytrain[j] ≥ 1 - ξ[j]))
    end
    
    optimize!(model)

    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = dot(PredFix,x) + PredRand
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end

function di_me_svm(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain]
    sorted_indices = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices, :]

    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    m,n = size(xtrain)
    
    try 
        global Z = Vector(xtrain[!,SF])
    catch
        global Z = Matrix(xtrain[!,SF])
    end
    means_z = Statistics.mean(Z, dims=1)

    zeh = zeros(m,length(SF))
    for p = 1 : length(SF)
        zeh[1:nct[1],p] = Z[1:nct[1],p] .- means_z[1,p]
    end
    for p = 1 : length(SF)
        for i = 2 : nc
            zeh[sum(nct[j] for j = 1 : i-1)+1:sum(nct[j] for j = 1 : i-1)+nct[i],p] = Z[sum(nct[j] for j = 1 : i-1)+1:sum(nct[j] for j = 1 : i-1)+nct[i],p] .- means_z[1,p]
        end
    end

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ) + dot(b,b))
    
    @constraint(model, [j=1:nct[1]], (dot(β, xtrain[j,:]) + b[1]) * ytrain[j] ≥ 1 - ξ[j])
    for i = 2 : nc
        @constraint(model, [j=sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]], ((dot(β, xtrain[j,:]) + b[i]) * ytrain[j] ≥ 1 - ξ[j]))
    end
    
    @constraint(model, [k=1:size(Z,2)], ((1/m)*sum(zeh[j,k]*(dot(β, xtrain[j,:]) + b[1]) for j = 1 : nct[1]) + (1/m)*sum(sum(zeh[j,k]*(dot(β, xtrain[j,:]) + b[i]) for j = sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]) for i = 2 : nc)) ≤ c)
    @constraint(model, [k=1:size(Z,2)], ((1/m)*sum(zeh[j,k]*(dot(β, xtrain[j,:]) + b[1]) for j = 1 : nct[1]) + (1/m)*sum(sum(zeh[j,k]*(dot(β, xtrain[j,:]) + b[i]) for j = sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]) for i = 2 : nc)) ≥ -c)

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = dot(PredFix,x) + PredRand
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end

function fnr_me_svm(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    xtrain = [group_id_train xtrain]
    xtrain = rename(xtrain,:x1 => :ClusterID)
    xtrain = sort(xtrain, :ClusterID)
    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    

    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    nct0 = zeros(Int64, nc, length(SF))
    nct1 = zeros(Int64, nc, length(SF))
    N02 = zeros(Int64,length(SF))
    N12 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        N02[i] = size(a0,1)
        N12[i] = size(a1,1)
        fa0 = findall(==(1), a0[:,end])
        fa1 = findall(==(1), a1[:,end])
        a0 = a0[fa0,:]
        a1 = a1[fa1,:]
        for num in a0.ClusterID
            nct0[num,i] += 1
        end
        for num in a1.ClusterID
            nct1[num,i] += 1
        end
        xtrain0[i] = a0[:,2:end-1]
        xtrain1[i] = a1[:,2:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]] 
    end
    xtrain = xtrain[:,2:end]
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ) + dot(b,b))
    
    @constraint(model, [j=1:nct[1]], (dot(β, xtrain[j,:]) + b[1]) * ytrain[j] ≥ 1 - ξ[j])
    for i = 2 : nc
        @constraint(model, [j=sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]], ((dot(β, xtrain[j,:]) + b[i]) * ytrain[j] ≥ 1 - ξ[j]))
    end
    
    
    @NLconstraint(model, [k=1:length(SF)], ((-N1[k]/m)*(sum(min(0,(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[1])) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,((sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[i]))) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0[k]/m)*(sum(min(0,(sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[1])) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[i])) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) ≤ c)   
    @NLconstraint(model, [k=1:length(SF)], ((-N1[k]/m)*(sum(min(0,(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[1])) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,((sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[i]))) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0[k]/m)*(sum(min(0,(sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[1])) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[i])) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) ≥ -c)   

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = dot(PredFix,x) + PredRand
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end

function fpr_me_svm(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    xtrain = [group_id_train xtrain]
    xtrain = rename(xtrain,:x1 => :ClusterID)
    xtrain = sort(xtrain, :ClusterID)
    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    
    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    nct0 = zeros(Int64, nc, length(SF))
    nct1 = zeros(Int64, nc, length(SF))
    N02 = zeros(Int64,length(SF))
    N12 = zeros(Int64,length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        N02[i] = size(a0,1)
        N12[i] = size(a1,1)
        fa0 = findall(==(1), a0[:,end])
        fa1 = findall(==(1), a1[:,end])
        a0 = a0[fa0,:]
        a1 = a1[fa1,:]
        for num in a0.ClusterID
            nct0[num,i] += 1
        end
        for num in a1.ClusterID
            nct1[num,i] += 1
        end
        xtrain0[i] = a0[:,2:end-1]
        xtrain1[i] = a1[:,2:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
        ytrain0[i] .= [y == 1 ? 1 : -1 for y in ytrain0[i]] 
        ytrain1[i] .= [y == 1 ? 1 : -1 for y in ytrain1[i]] 
    end
    xtrain = xtrain[:,2:end]
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ) + dot(b,b))
    
    @constraint(model, [j=1:nct[1]], (dot(β, xtrain[j,:]) + b[1]) * ytrain[j] ≥ 1 - ξ[j])
    for i = 2 : nc
        @constraint(model, [j=sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]], ((dot(β, xtrain[j,:]) + b[i]) * ytrain[j] ≥ 1 - ξ[j]))
    end
    
    @NLconstraint(model, [k=1:length(SF)], ((-N1[k]/m)*(sum(min(0,-(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[1])) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(-(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[i]))) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0[k]/m)*(sum(min(0,(-sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[1])) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(-sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[i])) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) ≤ c)   
    @NLconstraint(model, [k=1:length(SF)], ((-N1[k]/m)*(sum(min(0,-(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[1])) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(-(sum(β[g] * xtrain0[k][j,g] for g=1:n) + b[i]))) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0[k]/m)*(sum(min(0,(-sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[1])) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(-sum(β[g] * xtrain1[k][j,g] for g=1:n) + b[i])) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) ≥ -c)   

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = dot(PredFix,x) + PredRand
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end

function dm_me_svm(xtrain, ytrain, newdata, SF, c, group_id_train, group_id_newdata)
    categories = vcat(group_id_train,group_id_newdata)
    numeric = levelcode.(categories)
    group_id_train = numeric[1:length(group_id_train)]
    group_id_newdata = numeric[length(group_id_train)+1:length(group_id_newdata)+length(group_id_train)]
    if any(xtrain[:, 1] .!= 1.0) && any(xtrain[:, end] .!= 1.0)
        insertcols!(xtrain, 1, :Intercept => ones(size(xtrain,1)))
    end
    if any(newdata[:, 1] .!= 1.0) && any(newdata[:, end] .!= 1.0)
        insertcols!(newdata, 1, :Intercept => ones(size(newdata,1)))
    end
    ytrain = [y == 1 ? 1 : -1 for y in ytrain] 
    xtrain = [group_id_train xtrain]
    xtrain = rename(xtrain,:x1 => :ClusterID)
    xtrain = sort(xtrain, :ClusterID)
    nct = collect(values(sort(countmap(group_id_train))))
    nc = length(nct)
    

    xtrain0 = [DataFrame() for _ in 1:length(SF)]
    xtrain1 = [DataFrame() for _ in 1:length(SF)]
    ytrain0 = [Vector() for _ in 1:length(SF)]
    ytrain1 = [Vector() for _ in 1:length(SF)]
    N0 = zeros(Int64,length(SF))
    N1 = zeros(Int64,length(SF))
    nct0 = zeros(Int64, nc, length(SF))
    nct1 = zeros(Int64, nc, length(SF))
    xtraint = copy(xtrain)
    xtraint[!,:Labels] = ytrain

    for i = 1 : length(SF)
        a0,a1 = groupby(xtraint, SF[i])
        for num in a0.ClusterID
            nct0[num,i] += 1
        end
        for num in a1.ClusterID
            nct1[num,i] += 1
        end
        xtrain0[i] = a0[:,2:end-1]
        xtrain1[i] = a1[:,2:end-1]
        ytrain0[i] = a0[:,end]
        ytrain1[i] = a1[:,end]
        N0[i] = size(a0,1)
        N1[i] = size(a1,1)
    end
    xtrain = xtrain[:,2:end]
    m,n = size(xtrain)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])
    @variable(model, ξ[1:m] ≥ 0)
    
    @objective(model, Min, dot(β, β)/2 + sum(ξ) + dot(b,b))
    
    @constraint(model, [j=1:nct[1]], (dot(β, xtrain[j,:]) + b[1]) * ytrain[j] ≥ 1 - ξ[j])
    for i = 2 : nc
        @constraint(model, [j=sum(nct[r] for r = 1 : i-1)+1:sum(nct[r] for r = 1 : i-1)+nct[i]], ((dot(β, xtrain[j,:]) + b[i]) * ytrain[j] ≥ 1 - ξ[j]))
    end
    
    @constraint(model, [k=1:length(SF)], ((-N1/m)*(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[1]) * ((1+ytrain0[k][j])/2)*ytrain0[k][j]) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[i]) * ((1+ytrain0[k][j])/2)*ytrain0[k][j]) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0/m)*(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[1]) * ((1+ytrain1[k][j])/2)*ytrain1[k][j]) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[i]) * ((1+ytrain1[k][j])/2)*ytrain1[k][j]) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) .≤ c)   
    @constraint(model, [k=1:length(SF)], ((-N1/m)*(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[1]) * ((1+ytrain0[k][j])/2)*ytrain0[k][j]) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[i]) * ((1+ytrain0[k][j])/2)*ytrain0[k][j]) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0/m)*(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[1]) * ((1+ytrain1[k][j])/2)*ytrain1[k][j]) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[i]) * ((1+ytrain1[k][j])/2)*ytrain1[k][j]) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) .≥ -c)

    @constraint(model, [k=1:length(SF)], ((-N1/m)*(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[1]) * ((1-ytrain0[k][j])/2)*ytrain0[k][j]) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[i]) * ((1-ytrain0[k][j])/2)*ytrain0[k][j]) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0/m)*(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[1]) * ((1-ytrain1[k][j])/2)*ytrain1[k][j]) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[i]) * ((1-ytrain1[k][j])/2)*ytrain1[k][j]) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) .≤ c)   
    @constraint(model, [k=1:length(SF)], ((-N1/m)*(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[1]) * ((1-ytrain0[k][j])/2)*ytrain0[k][j]) for j = 1 : nct0[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain0[k][j,:]) + b[i]) * ((1-ytrain0[k][j])/2)*ytrain0[k][j]) for j = sum(nct0[:,k][r] for r = 1 : i-1)+1:sum(nct0[:,k][r] for r = 1 : i-1)+nct0[:,k][i]; init=0) for i = 2 : nc; init=0))) + ((N0/m)*(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[1]) * ((1-ytrain1[k][j])/2)*ytrain1[k][j]) for j = 1 : nct1[:,k][1]; init=0) + sum(sum(min(0,(dot(β, xtrain1[k][j,:]) + b[i]) * ((1-ytrain1[k][j])/2)*ytrain1[k][j]) for j = sum(nct1[:,k][r] for r = 1 : i-1)+1:sum(nct1[:,k][r] for r = 1 : i-1)+nct1[:,k][i]; init=0) for i = 2 : nc; init=0))) .≥ -c)

    optimize!(model)
    
    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    sorted_indices2 = sortperm(group_id_newdata)
    newdata = newdata[sorted_indices2, :]
    m1(x,PredFix,PredRand) = dot(PredFix,x) + PredRand
    nct2 = collect(values(sort(countmap(group_id_newdata))))
    
    ly = size(newdata,1)
    prob_newdata = zeros(ly)
    k=1
    group = zeros(nc)
    for i = 1 : nc
        group[i] = sum(nct2[j] for j = 1 : i)
    end

    for i = 1 : ly
        prob_newdata[i] = m1(newdata[i,:],PredFix,PredRand[k])
        if i > group[1]
            k += 1
            group = group[2:end]
        end
    end

    sorted_indices3 = sortperm(group_id_train)
    xtrain = xtrain[sorted_indices3, :]
    nct3 = collect(values(sort(countmap(group_id_train))))
    
    ly2 = size(xtrain,1)
    prob_train = zeros(ly2)
    k=1
    groupt = zeros(nc)
    for i = 1 : nc
        groupt[i] = sum(nct3[j] for j = 1 : i)
    end

    for i = 1 : ly2
        prob_train[i] = m1(xtrain[i,:],PredFix,PredRand[k])
        if i > groupt[1]
            k += 1
            groupt = groupt[2:end]
        end
    end

    return prob_train, prob_newdata
end





#Post-processing phase
function id_post(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost)  
    Prediction = zeros(length(prob_newdata))
    for i = 1 : length(prob_newdata)
        if prob_newdata[i] < 0.5
            Prediction[i] = -1
        else 
            Prediction[i] = 1
        end
    end
    return Prediction
end

function di_post(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost) 
    xtrain2 = copy(xtrain)
    tests = collect(0.01:0.01:1)
    DIs = zeros(length(tests))
    ACCs = zeros(length(tests))
    for k = 1 : length(tests)
        i = tests[k]
        Prediction = zeros(length(prob_train))
        for j = 1 : length(prob_train)
            if prob_train[j] < i
                Prediction[j] = -1
            else 
                Prediction[j] = 1
            end
        end
        DIs[k] = disparate_impact_metric(xtrain2, Prediction, SFpost)
        ACCs[k] = final_metrics(ytrain, Prediction)[1] 
    end
    
    cvID = ACCs[50,:]*0.95
    A = [ACCs DIs]
    fa = findall(x -> x > cvID[1], A[:,1])
    best1 =  argmax(filter(!isnan, (ACCs[fa].-DIs[fa])))
    best = fa[best1]


    Prediction = zeros(length(prob_newdata))
    for i = 1 : size(newdata,1)
        if prob_newdata[i] < tests[best]
            Prediction[i] = -1
        else 
            Prediction[i] = 1
        end
    end

    return Prediction
end

function fnr_post(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost) 
    tests = collect(0.01:0.01:1)
    DIs = zeros(length(tests))
    ACCs = zeros(length(tests))
    for k = 1 : length(tests)
        i = tests[k]
        Prediction = zeros(length(prob_train))
        for j = 1 : length(prob_train)
            if prob_train[j] ≤ i
                Prediction[j] = -1
            else 
                Prediction[j] = 1
            end
        end
        DIs[k] = false_negative_rate_metric(xtrain, ytrain, Prediction, SFpost)
        ACCs[k] = final_metrics(ytrain, Prediction)[1] 
    end
    
    cvID = ACCs[50,:]*0.95
    A = [ACCs DIs]
    fa = findall(x -> x > cvID[1], A[:,1])
    best1 =  argmax(filter(!isnan, (ACCs[fa].-DIs[fa])))
    best = fa[best1]

    Prediction = zeros(length(prob_newdata))
    for i = 1 : size(newdata,1)
        if prob_newdata[i] < tests[best]
            Prediction[i] = -1
        else 
            Prediction[i] = 1
        end
    end

    return Prediction
end

function fpr_post(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost) 
    tests = collect(0.01:0.01:1)
    DIs = zeros(length(tests))
    ACCs = zeros(length(tests))
    for k = 1 : length(tests)
        i = tests[k]
        Prediction = zeros(length(prob_train))
        for j = 1 : length(prob_train)
            if prob_train[j] ≤ i
                Prediction[j] = -1
            else 
                Prediction[j] = 1
            end
        end
        DIs[k] = false_positive_rate_metric(xtrain, ytrain, Prediction, SFpost)
        ACCs[k] = final_metrics(ytrain, Prediction)[1] 
    end
    
    cvID = ACCs[50,:]*0.95
    A = [ACCs DIs]
    fa = findall(x -> x > cvID[1], A[:,1])
    best1 =  argmax(filter(!isnan, (ACCs[fa].-DIs[fa])))
    best = fa[best1]
    Prediction = zeros(length(prob_newdata))
    for i = 1 : size(newdata,1)
        if prob_newdata[i] < tests[best]
            Prediction[i] = -1
        else 
            Prediction[i] = 1
        end
    end

    return Prediction
end

function tnr_post(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost) 
    tests = collect(0.01:0.01:1)
    DIs = zeros(length(tests))
    ACCs = zeros(length(tests))
    for k = 1 : length(tests)
        i = tests[k]
        Prediction = zeros(length(prob_train))
        for j = 1 : length(prob_train)
            if prob_train[j] ≤ i
                Prediction[j] = -1
            else 
                Prediction[j] = 1
            end
        end
        DIs[k] = true_negative_rate_metric(xtrain, ytrain, Prediction, SFpost)
        ACCs[k] = final_metrics(ytrain, Prediction)[1] 
    end
    
    cvID = ACCs[50,:]*0.95
    A = [ACCs DIs]
    fa = findall(x -> x > cvID[1], A[:,1])
    best1 =  argmax(filter(!isnan, (ACCs[fa].-DIs[fa])))
    best = fa[best1]
    Prediction = zeros(length(prob_newdata))
    for i = 1 : size(newdata,1)
        if prob_newdata[i] < tests[best]
            Prediction[i] = -1
        else 
            Prediction[i] = 1
        end
    end

    return Prediction
end

function tpr_post(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost) 
    tests = collect(0.01:0.01:1)
    DIs = zeros(length(tests))
    ACCs = zeros(length(tests))
    for k = 1 : length(tests)
        i = tests[k]
        Prediction = zeros(length(prob_train))
        for j = 1 : length(prob_train)
            if prob_train[j] ≤ i
                Prediction[j] = -1
            else 
                Prediction[j] = 1
            end
        end
        DIs[k] = true_positive_rate_metric(xtrain, ytrain, Prediction, SFpost)
        ACCs[k] = final_metrics(ytrain, Prediction)[1] 
    end
    
    cvID = ACCs[50,:]*0.95
    A = [ACCs DIs]
    fa = findall(x -> x > cvID[1], A[:,1])
    best1 =  argmax(filter(!isnan, (ACCs[fa].-DIs[fa])))
    best = fa[best1]
    Prediction = zeros(length(prob_newdata))
    for i = 1 : size(newdata,1)
        if prob_newdata[i] < tests[best]
            Prediction[i] = -1
        else 
            Prediction[i] = 1
        end
    end

    return Prediction
end

function dm_post(prob_train,prob_newdata,xtrain,ytrain,newdata,SFpost) 
    tests = collect(0.01:0.01:1)
    DIs = zeros(length(tests))
    ACCs = zeros(length(tests))
    for k = 1 : length(tests)
        i = tests[k]
        Prediction = zeros(length(prob_train))
        for j = 1 : length(prob_train)
            if prob_train[j] ≤ i
                Prediction[j] = -1
            else 
                Prediction[j] = 1
            end
        end
        DIs[k] = disparate_mistreatment_metric(xtrain, ytrain, Prediction, SFpost)
        ACCs[k] = final_metrics(ytrain, Prediction)[1] 
    end
    
    cvID = ACCs[50,:]*0.95
    A = [ACCs DIs]
    fa = findall(x -> x > cvID[1], A[:,1])
    best1 =  argmax(filter(!isnan, (ACCs[fa].-DIs[fa])))
    best = fa[best1]
    Prediction = zeros(length(prob_newdata))
    for i = 1 : size(newdata,1)
        if prob_newdata[i] < tests[best]
            Prediction[i] = -1
        else 
            Prediction[i] = 1
        end
    end

    return Prediction
end





#Fair Metrics
function final_metrics(ynewdata, predictions)    
    ynewdata = [y == 1 ? 1 : 0 for y in ynewdata] 
    predictions = [y == 1 ? 1 : 0 for y in predictions]
    cm1 = EvalMetrics.ConfusionMatrix(ynewdata, predictions)
    ac = EvalMetrics.accuracy(cm1)
    fpr = EvalMetrics.false_positive_rate(cm1)
    fnr = EvalMetrics.false_negative_rate(cm1)
    tpr = EvalMetrics.true_positive_rate(cm1)
    tnr = EvalMetrics.true_negative_rate(cm1)
    rc = EvalMetrics.recall(cm1)
    TP = EvalMetrics.true_positive(cm1)
    FP = EvalMetrics.false_positive(cm1)
    TN = EvalMetrics.true_negative(cm1)
    FN = EvalMetrics.false_negative(cm1)

    return ac, fpr, fnr, tpr, tnr, rc, TP, FP, TN, FN
end

function disparate_impact_metric(newdata, predictions, SF)
    newdata2 = copy(newdata)
    if any(newdata2[:, 1] .!= 1.0) && any(newdata2[:, end] .!= 1.0)
        insertcols!(newdata2, 1, :Intercept => ones(size(newdata2,1)))
    end
    predictions2 = [y == 1 ? 1 : 0 for y in predictions] 
    newdata2[!,:Labels] = predictions2
    newdata2 = sort(newdata2, [SF])
    split = findfirst(==(1), newdata2[!, SF][:,1])
    newdata0 = newdata2[1:split-1,:]
    newdata1 = newdata2[split:end,:]
    k11 = size(newdata1,1)
    k10 = size(newdata0,1)

    previsao_1 = newdata1[:,end]
    c1 = sum(previsao_1)/k11

    previsao_0 = newdata0[:,end]
    c0 = sum(previsao_0)/k10

    Dis_Impact = 1-min(c1/c0, c0/c1)
    return Dis_Impact 
end

function false_negative_rate_metric(newdata, ynewdata, predictions, SF)
    newdata2 = copy(newdata)
    if any(newdata2[:, 1] .!= 1.0) && any(newdata2[:, end] .!= 1.0)
        insertcols!(newdata2, 1, :Intercept => ones(size(newdata2,1)))
    end
    ynewdata2 = [y == 1 ? 1 : 0 for y in ynewdata]
    predictions2 = [y == 1 ? 1 : 0 for y in predictions] 
    newdata2[!,:Labels] = ynewdata2
    Data_Test = newdata2
    Data_Test = hcat(Data_Test, predictions2,makeunique=true)
    newdata2 = sort(Data_Test, [SF])
    split = findfirst(==(1), newdata2[!, SF][:,1])

    ynewdata0 = newdata2[1:split-1,end-1]
    ynewdata1 = newdata2[split:end,end-1]
    Pred0 = newdata2[1:split-1,end]
    Pred1 = newdata2[split:end,end]

    ac, fpr, fnr, tpr, tnr, rc, TP, FP, TN, FN = final_metrics(ynewdata2, predictions2)
    ac0, fpr0, fnr0, tpr0, tnr0, rc0, TP0, FP0, TN0, FN0 = final_metrics(ynewdata0, Pred0)
    ac1, fpr1, fnr1, tpr1, tnr1, rc1, TP1, FP1, TN1, FN1 = final_metrics(ynewdata1, Pred1)

    FNR_Metric = abs(fnr1-fnr0)
    return FNR_Metric 
end

function false_positive_rate_metric(newdata, ynewdata, predictions, SF)
    newdata2 = copy(newdata)
    if any(newdata2[:, 1] .!= 1.0) && any(newdata2[:, end] .!= 1.0)
        insertcols!(newdata2, 1, :Intercept => ones(size(newdata2,1)))
    end
    ynewdata2 = [y == 1 ? 1 : 0 for y in ynewdata]
    predictions2 = [y == 1 ? 1 : 0 for y in predictions] 
    newdata2[!,:Labels] = ynewdata2
    Data_Test = newdata2
    Data_Test = hcat(Data_Test, predictions2, makeunique=true)
    newdata2 = sort(Data_Test, [SF])
    split = findfirst(==(1), newdata2[!, SF][:,1])

    ynewdata0 = newdata2[1:split-1,end-1]
    ynewdata1 = newdata2[split:end,end-1]
    Pred0 = newdata2[1:split-1,end]
    Pred1 = newdata2[split:end,end]

    ac, fpr, fnr, tpr, tnr, rc, TP, FP, TN, FN = final_metrics(ynewdata2, predictions2)
    ac0, fpr0, fnr0, tpr0, tnr0, rc0, TP0, FP0, TN0, FN0 = final_metrics(ynewdata0, Pred0)
    ac1, fpr1, fnr1, tpr1, tnr1, rc1, TP1, FP1, TN1, FN1 = final_metrics(ynewdata1, Pred1)

    FPR_Metric = abs(fpr1-fpr0)
    return FPR_Metric 
end

function true_negative_rate_metric(newdata, ynewdata, predictions, SF)
    newdata2 = copy(newdata)
    if any(newdata2[:, 1] .!= 1.0) && any(newdata2[:, end] .!= 1.0)
        insertcols!(newdata2, 1, :Intercept => ones(size(newdata2,1)))
    end
    ynewdata2 = [y == 1 ? 1 : 0 for y in ynewdata]
    predictions2 = [y == 1 ? 1 : 0 for y in predictions] 
    newdata2[!,:Labels] = ynewdata2
    Data_Test = newdata2
    Data_Test = hcat(Data_Test, predictions2,makeunique=true)
    newdata2 = sort(Data_Test, [SF])
    split = findfirst(==(1), newdata2[!, SF][:,1])

    ynewdata0 = newdata2[1:split-1,end-1]
    ynewdata1 = newdata2[split:end,end-1]
    Pred0 = newdata2[1:split-1,end]
    Pred1 = newdata2[split:end,end]

    ac, fpr, fnr, tpr, tnr, rc, TP, FP, TN, FN = final_metrics(ynewdata2, predictions2)
    ac0, fpr0, fnr0, tpr0, tnr0, rc0, TP0, FP0, TN0, FN0 = final_metrics(ynewdata0, Pred0)
    ac1, fpr1, fnr1, tpr1, tnr1, rc1, TP1, FP1, TN1, FN1 = final_metrics(ynewdata1, Pred1)

    TNR_Metric = abs(tnr1-tnr0)
    return TNR_Metric 
end

function true_positive_rate_metric(newdata, ynewdata, predictions, SF)
    newdata2 = copy(newdata)
    if any(newdata2[:, 1] .!= 1.0) && any(newdata2[:, end] .!= 1.0)
        insertcols!(newdata2, 1, :Intercept => ones(size(newdata2,1)))
    end
    ynewdata2 = [y == 1 ? 1 : 0 for y in ynewdata]
    predictions2 = [y == 1 ? 1 : 0 for y in predictions] 
    newdata2[!,:Labels] = ynewdata2
    Data_Test = newdata2
    Data_Test = hcat(Data_Test, predictions2,makeunique=true)
    newdata2 = sort(Data_Test, [SF])
    split = findfirst(==(1), newdata2[!, SF][:,1])

    ynewdata0 = newdata2[1:split-1,end-1]
    ynewdata1 = newdata2[split:end,end-1]
    Pred0 = newdata2[1:split-1,end]
    Pred1 = newdata2[split:end,end]

    ac, fpr, fnr, tpr, tnr, rc, TP, FP, TN, FN = final_metrics(ynewdata2, predictions2)
    ac0, fpr0, fnr0, tpr0, tnr0, rc0, TP0, FP0, TN0, FN0 = final_metrics(ynewdata0, Pred0)
    ac1, fpr1, fnr1, tpr1, tnr1, rc1, TP1, FP1, TN1, FN1 = final_metrics(ynewdata1, Pred1)

    TPR_Metric = abs(tpr1-tpr0)
    return TPR_Metric 
end

function disparate_mistreatment_metric(newdata, ynewdata, predictions, SF)
    newdata2 = copy(newdata)
    if any(newdata2[:, 1] .!= 1.0) && any(newdata2[:, end] .!= 1.0)
        insertcols!(newdata2, 1, :Intercept => ones(size(newdata2,1)))
    end
    ynewdata2 = [y == 1 ? 1 : 0 for y in ynewdata]
    predictions2 = [y == 1 ? 1 : 0 for y in predictions] 
    
    newdata2[!,:Labels] = ynewdata2
    Data_Test = newdata2

    Data_Test = hcat(Data_Test, predictions2,makeunique=true)
    newdata2 = sort(Data_Test, [SF])
    split = findfirst(==(1), newdata2[!, SF][:,1])

    ynewdata0 = newdata2[1:split-1,end-1]
    ynewdata1 = newdata2[split:end,end-1]
    Pred0 = newdata2[1:split-1,end]
    Pred1 = newdata2[split:end,end]

    ac, fpr, fnr, tpr, tnr, rc, TP, FP, TN, FN = final_metrics(ynewdata2, predictions2)
    ac0, fpr0, fnr0, tpr0, tnr0, rc0, TP0, FP0, TN0, FN0 = final_metrics(ynewdata0, Pred0)
    ac1, fpr1, fnr1, tpr1, tnr1, rc1, TP1, FP1, TN1, FN1 = final_metrics(ynewdata1, Pred1)

    DM_Metric = (abs(fnr1-fnr0) + abs(fpr1-fpr0))/2
    return DM_Metric 
end





end # module FairML
