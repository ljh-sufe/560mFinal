using DataFrames, CSV, ShiftedArrays, Statistics, Dates, ScikitLearn, PrettyPrinting
using PyPlot, StableRNGs, MLJ
import ShiftedArrays: lead, lag
import DataFrames as DF, Statistics
import StatsModels as SM
import LinearAlgebra as LA

df = DataFrame(CSV.File("data/train_timeseries/train_timeseries.csv"))
soil_df = DataFrame(CSV.File("data/soil_data.csv"))


static_data_cols = sort(filter(x-> x ∉ ["soil","lat","lon"], names(soil_df)))

df=dropmissing(df) # drop nan values

## generate target: the drought level of next week
df[!,"target"] = combine(groupby(df, :fips), :score => lead)[:,:score_lead]
df[:, ["fips","score","target"]]

## handling with Dates 
df[!, "MONTH"] = Dates.month.(df[:, "date"])


time_data_cols = sort(["PRECTOT","PS","QV2M","T2M","T2MDEW","T2MWET",
"T2M_MAX","T2M_MIN","T2M_RANGE","TS","WS10M","WS10M_MAX",
"WS10M_MIN","WS10M_RANGE","WS50M","WS50M_MAX","WS50M_MIN","WS50M_RANGE","MONTH","score"])


## train
df=dropmissing(df) # drop nan values

train_x = df[:, time_data_cols]
train_y = df[:, :target]

monthDummy = transpose(unique(train_x[:,:MONTH]) .== permutedims(train_x[:,:MONTH]))
monthDummy = LA.convert(Array{Float64}, monthDummy)
monthDummy = DF.DataFrame(monthDummy[:, 2:12], map(x->"MONTH" * string(x), 2:12))

train_x = DF.select!(DF.hcat(monthDummy, train_x), Not(:MONTH))


using MLJ
import MLJ:learning_curve, fit! as MLJfit

MLJ.models(matching(train_x, train_y))

## decision tree model
Tree = @load DecisionTreeRegressor pkg=DecisionTree
tree = Tree()

mach = machine(tree, train_x, train_y)
MLJfit(mach)
yhat = MLJ.predict(mach, train_x);

MLJ.mav(yhat, train_y)


## 
RidgeRegressor = @load RidgeRegressor pkg=MultivariateStats

ridge = RidgeRegressor()
mach = machine(ridge, train_x, train_y)
MLJfit(mach)
yhat = MLJ.predict(mach, train_x);
