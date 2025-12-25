## AI Application Boost with RAPIDS GPU Acceleration
- Instructor: Jones Granatyr, Gabriel Alves

## Section 1: Introduction

### 1. Course content
- Introduction to CPU & GPU
- Pandas vs cuDF
- sklearn vs cuML
- Dask + cuDF + cuML

### 2. CPU vs GPU
- RAPIDS: a suite of open source libraries that enable end-to-end execution of data science and ML pipelines entirely on GPUs
  - https://rapids.ai/
- Benefits of RAPIDS
  - Easy and familiar integration
  - Productivity boost
  - Increase in speed and accuracy
  - Cost reduction
  - Open source 
- RAPIDS works connected with Apache Arrow, inside GPU memory
  - https://arrow.apache.org/
  - Apache Arrow provides an efficient and interoperable columnar memory format, transforming a table into columnar data
- RAPIDS APIs 
  - cuDF
  - cuML
  - cuGraph
  - cuSpatial
  - cuSignal
  - cuCIM
- RAPIDS can be integrated with
  - Dask: workloads across several GPUs
  - Apache Spark
  - Dask SQL
  - XGBoost

### 3. GPU and CUDA

### 4. RAPIDS

### 5. Course materials

## Section 2: cuDF

### 6. cuDF - intuition
- cuDF
- cuML
- cuPy

### 7. Installation
- https://docs.rapids.ai/install/
  - Select the current configuration then it will generate an appropriate command:
  - conda create -n rapids-25.12 -c rapidsai -c conda-forge  rapids=25.12 python=3.11 'cuda-version>=12.2,<=12.9'
- source ~/sw_local/anaconda3/2023.07/etc/profile.d/conda.sh
- conda activate rapids-25.12
- python3
- If Jupyter notebook/lab doesn't work, reinstall: conda install jupyterlab

### 8. Pandas and cuDF
```py
import pandas as pd
print(pd.__version__)
df = pd.DataFrame()
df['id'] = [ 0,1,2,2,3,3,3]
df['val']= [float(i+10) for i in range(7)]
df
print(type(df), df['val'].sum()) # <class 'pandas.core.frame.DataFrame'> 91.0
##
## Reproducing using cudf
import cudf
print(cudf.__version__)
cdf = cudf.DataFrame()
cdf['id'] = [ 0,1,2,2,3,3,3]
cdf['val']= [float(i+10) for i in range(7)]
cdf
print(type(cdf), cdf['val'].sum()) # <class 'cudf.core.dataframe.DataFrame'> 91.0>
cdf.sort_values(by='b')
cdf.loc[3:5, ['a', 'b']]
#  	a	b
# 3	3	6
# 4	4	5
# 5	5	4

``` 

### 9. Basic commands 1
```py
s = cudf.Series([1,2,None,3, 4])
cdf = cudf.DataFrame({'a': list(range(10)),
                      'b': list(reversed(range(10))),
                      'c': list(range(10))})
cdf
##
# Conversion from pandas to cudf
df_p = pd.DataFrame({'a':[0,1,2,3], 'b':[0.1, 0.2, None, 0.3]})
df_c = cudf.DataFrame.from_pandas(df_p)
df_c
## .from_pands() will be deprecated
```

### 10. Basic commands 2
```py
cdf[cdf['b'] > 5]
cdf.query("b==7")
```
- cuPy as an alternative of numpy
```py
import numpy as np
np.arange(10) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
s = cudf.Series(np.arange(10).astype(np.float32))
print(s.mean(), s.var(), s.std()) # 4.5 9.166667 3.0276504
```
- cdf.describe(): yields the statistics of the DataFrame

### 11. Basic commands 3
```py
s = cudf.Series(['A','B','C', 'Rapids', None, 'Cat', 'Dog'])
print(s.str.upper())
print(s.str.lower())
print(s.str.byte_count())
# 0       1
# 1       1
# 2       1
# 3       6
# 4    <NA>
# 5       3
# 6       3
# dtype: int32
print(s.str.contains('C|cat'))
# 0    False
# 1    False
# 2     True
# 3    False
# 4     <NA>
# 5     True
# 6    False
# dtype: bool
s = cudf.Series([1,2,7,3,4])
def add_10(num):
  return num + 10
s.apply(add_10)
#0    11
#1    12
#2    17
#3    13
#4    14
#dtype: int64
```

### 12. Basic commands 4
```py
cdf['ag_col1'] = [1 if x%2 == 0 else 0 for x in range(len(cdf))]
cdf['ag_col2'] = [1 if x%3 == 0 else 0 for x in range(len(cdf))]
cdf
#	a	b	c	ag_col1	ag_col2
#0	0	9	0	1	1
#1	1	8	1	0	0
#2	2	7	2	1	0
#3	3	6	3	0	1
#4	4	5	4	1	0
#5	5	4	5	0	0
#6	6	3	6	1	1
#7	7	2	7	0	0
#8	8	1	8	1	0
#9	9	0	9	0	1
cdf.groupby('ag_col1').sum()
#           a	  b	  c	  ag_col2
# ag_col1				
#       1 	20	25	20	2
#       0	  25	20	25	2
cdf.groupby(['ag_col1','ag_col2']).agg({'a':'max', 'b': 'min'})
#		             a	b
#ag_col1	ag_col2		
#      1        0	8	1
#               1	6	3
#      0	      0	7	2
#               1	9	0
date_cdf = cudf.DataFrame()
date_cdf['date'] = pd.date_range('2024-01-29', periods=72, freq='W')
date_cdf['value'] = np.random.sample(len(date_cdf))
date_cdf
date_search = cudf.to_datetime('2024-02-20')
date_cdf.loc[date_cdf['date'] < date_search]
#         date	value
#0	2024-02-04	0.390281
#1	2024-02-11	0.688477
#2	2024-02-18	0.433442
```

### 13. Integration with cuPy
- Conversion of cudf into cupy
  - Using dlpack()
  - Using .values
  - Using .to_cupy()
```py
import cupy as cp
# cudf into cupy
nums = 10000
cdf = cudf.DataFrame({'a':range(nums), 'b':range(500,nums+500), 'c':range(1000, nums+1000)})
cdf
array_cupy = cp.from_dlpack(cdf.to_dlpack())
array_cupy
array_cupy2 = cdf.values
array_cupy2
array_cupy3 = cdf.to_cupy()
```
- Converting from cuPy to cudf
```py
df_cudf = cudf.DataFrame(array_cupy)
df_cudf
```

### 14. Other data convertions
```py
print(type(cdf)) # <class 'cudf.core.dataframe.DataFrame'>
df_pandas = cdf.to_pandas()
type(df_pandas) # pandas.core.frame.DataFrame
df_numpy = cdf.to_numpy()
df_a = cdf['a']
type(df_a) # cudf.core.series.Series
df_a_numpy = df_a.to_numpy()
type(df_a_numpy) # numpy.ndarray
cdf.to_csv('cudf_ex.csv', index=False)
# 
cdf_loaded = cudf.read_csv('cudf_ex.csv') # faster than pd.read_csv()
cdf_loaded
```

### 15. User defined functions 1
- Ref: https://numba.pydata.org/numba-doc/dev/cuda/cudapysupported.html
- Not supported:
  - Exception handling of try block
  - with block
  - Comprehension
  - Generator
- UDF in series
  - A standard python funciton with cudf.Series.apply
  - Using Numba Kernel: forall
```py
s = cudf.Series([1,2,7,3])
s.apply(add_10)
def add(x,num):
  return x+num
s.apply(add,args=(10,))
#0    11
#1    12
#2    17
#3    13
#dtype: int64
s = cudf.Series(['A', 'B', 'C', 'Rapids', None, 'Cat', 'Dog'])
def udf_string(st):
  if len(st)>0:
    if st.startswith("C"):
      return 1
    elif "d" in st:
      return 2
    else: 
      return -1
  else:
    return 0
res = s.apply(udf_string)
print(res)
#0   -1
#1   -1
#2    1
#3    2
#4    0
#5    1
#6   -1
#dtype: int64
from cudf.datasets import randomdata
cdf = randomdata(nrows=5,dtypes={"a":int, "b":int, "c": int})
cdf
#     	a	   b	  c
#0	1012	1037	998
#1	1030	959	1016
#2	967	974	1003
#3	1045	975	954
#4	1006	1031	997
from numba import cuda
@cuda.jit
def mult(in_col, out_col, num):
  i = cuda.grid(1)
  if i< in_col.size:
    out_col[i] = in_col[i] *num
size=len(cdf["a"])
cdf["e"] = 0.0
mult.forall(size)(cdf["a"], cdf["e"], 10.0)
print(cdf)
#      a     b     c        e
#0  1012  1037   998  10120.0
#1  1030   959  1016  10300.0
#2   967   974  1003   9670.0
#3  1045   975   954  10450.0
#4  1006  1031   997  10060.0    
```

### 16. User defined functions 2
```py
def udf_add(row):
  return row["a"] + row["b"]
cdf = cudf.DataFrame({"a":[1,2,3,4], "b":[5,6, cudf.NA,8]})
cdf
cdf.apply(udf_add, axis=1)
#0       6
#1       8
#2    <NA>  <--- operation is not allowed, and becomes NA
#3      12
#dtype: int64
def udf_add(row):
  x = row["a"]
  if x is cudf.NA:
    return 0
  else:
    return x+1
cdf = cudf.DataFrame({"a":[1, cudf.NA,3]})  
cdf.apply(udf_add, axis=1)
#0    2
#1    0
#2    4
#dtype: int64
def f(row):
  return row["a"] * (row["b"] + (row["c"]/row["d"])) % row["e"]
cdf = cudf.DataFrame(
  {
    "a":[1,2,3],
    "b":[4,5,6],
    "c":[7,7,cudf.NA],
    "d":[8,9,1],
    "e":[7,1,6]
  }
)
print(cdf)
cdf.apply(f,axis=1)
#   a  b     c  d  e
#0  1  4     7  8  7
#1  2  5     7  9  1
#2  3  6  <NA>  1  6
#
# 0          4.875
#1    0.555555556
#2           <NA>
#dtype: float64
```
- In cudf, apply_rows() is deprecated

### 17. User defined functions 3
```py
s = cudf.Series([1.0, 2,3,4,7])
cp_array = cp.asarray(s)
type(cp_array) # cupy.ndarray
@cuda.jit
def mult_5(x,out):
  i = cuda.grid(1)
  if i < x.size:
    out[i] = x[i] * 5
res = cudf.Series(cp.zeros(len(s),dtype="int32"))    
res # all zero
mult_5.forall(s.shape[0])(s,res)
res
#0     5
#1    10
#2    15
#3    20
#4    35
#dtype: int32
```

### 18. Performance comparison 1
```py
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import os
import time
import timeit
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
num_rows= 10_000_000
df_pandas = pd.DataFrame(
  {
    "nums": np.random.randint(-1000,1000, num_rows, dtype="int64"),
    "companies": np.random.choice(["Google", "Amazon", "AWS", "Apple", "Meta"])
  }
)
%timeit
df_cudf = cudf.from_pandas(df_pandas)
def timeit_pandas_cudf(pd_obj,cudf_obj,func,**kwargs):
  t_pandas = timeit.timeit(lambda: func(pd_obj), **kwargs)
  t_cudf = timeit.timeit(lambda: func(cudf_obj), **kwargs)
  return t_pandas, t_cudf
```

### 19. Performance comparison 2
```py
pandas_concat = timeit.timeit(lambda: pd.concat([df_pandas,df_pandas]), number=30) # took 7sec
cudf_concat = timeit.timeit(lambda: cudf.concat([df_cudf,df_cudf]), number=30) # took 0.7sec
pd_groupy,cudf_groupby=timeit_pandas_cudf(df_pandas, df_cudf, lambda df: df.groupby("companies").agg(["min", "max", "mean"]), number=30)
pd_groupy, cudf_groupby # (19.27945284798625, 0.3889643340080511)
```

### 20. Performance comparison 3
```py
num_rows = 10_000_000
pandas_s = pd.Series(np.random.choice(["Googoles", "Amazon", "Apple", "MS", "Netflix"], size=num_rows))
cudf_s = cudf.from_pandas(pandas_s)
type(cudf_s) # cudf.core.series.Series
#
pd_upper, cudf_upper = timeit_pandas_cudf(pandas_s, cudf_s, lambda s: s.str.upper(), number=20)
pd_upper, cudf_upper # (27.62193874598597, 0.2612214690016117)
pd_contains, cudf_contains = timeit_pandas_cudf(pandas_s, cudf_s, lambda s: s.str.contains(r"[0-9][a-z]"), number=20)
pd_contains, cudf_contains # (35.71463911398314, 0.9328743270016275)
```

## Section 3: cuML

### 21. cuML - intution
- Based on scikit-learn, GPU-accelerated ML algorithms
- 10-50x speed-up than CPU

### 22. Preparing the environment
```py
import numpy as np
import pandas as pd
import cudf
import cuml
import os
import time
import timeit
import matplotlib.pyplot as plt
num = 10000
w = 2.0
x = np.random.normal(size=(num))
x
b = 1.0
y = w*x + b
noise = np.random.normal(scale=2.0, size=num)
y_noisy = y + noise
plt.scatter(x,y_noisy, label="Data")
```

### 23. Regression with scikit-learn
```py
import sklearn
from sklearn.linear_model import LinearRegression
print(sklearn.__version__)
linear_regression = LinearRegression()
#linear_regression.fit(x,y_noisy) # not working due to dim mismatching
linear_regression.fit(np.expand_dims(x,1),y_noisy)
inputs = np.linspace(start=-5,stop=5,num=1000000)
outputs = linear_regression.predict(np.expand_dims(inputs,1))
```

### 24. Regression with cuML
```py
cdf = cudf.DataFrame({"x":x, "y": y_noisy})
import cuml
print(cuml.__version__)
from cuml.linear_model import LinearRegression as LinearRegressionGPU
import cupy as cp
linear_regression_gpu = LinearRegressionGPU()
linear_regression_gpu.fit(cp.expand_dims(cp.array(cdf["x"]),1), cp.array(y_noisy))
df_cudf = cudf.DataFrame({'inputs':inputs})
outputs_gpu = linear_regression_gpu.predict(df_cudf[["inputs"]])
plt.scatter(x,y_noisy, label="data")
plt.plot(x,y, color="black", label="true relatin")
plt.plot(inputs, outputs, color="red", label="predictions")
plt.plot(inputs, outputs_gpu.get(), color="green", label="gpu_predictions")
```
- In plt.plot(), note that `outputs_gpu.get()` is necessary

### 25. Ridge regression
```py
from sklearn import datasets
diabetes = datasets.load_diabetes()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)
X_train.shape, y_train.shape # ((353, 10), (353,))
X_test.shape, y_test.shape # ((89, 10), (89,))
from cuml import Ridge as cuRidge
from sklearn.linear_model import Ridge as skRidge
# cpu
alpha = np.array([1.0])
fit_intercept = True
ridge = skRidge(alpha = alpha, fit_intercept=fit_intercept, solver="cholesky")
ridge.fit(X_train, y_train)
# gpu
X_train_cp = cp.array(X_train)
y_train_cp = cp.array(y_train)
cuml_ridge = cuRidge(alpha = alpha, fit_intercept=fit_intercept, solver="auto") # cholesky not supported
cuml_ridge.fit(X_train_cp, y_train_cp)
print('sklearn: ' + str(ridge.score(X_test,y_test)))
print('cuml: ' + str(cuml_ridge.score(X_test,y_test)))
```
- Same score of 0.318574

### 26. Parameter tuning
```py
from sklearn.model_selection import GridSearchCV
params = {'alpha': np.logspace(-3,-1, 10)}
grid = GridSearchCV(ridge, params, scoring="r2")
grid.fit(X_train, y_train)
grid.best_params_ # {'alpha': np.float64(0.007742636826811269)}
ridge = skRidge(alpha = grid.best_params_["alpha"], fit_intercept=fit_intercept, solver="cholesky")
ridge.fit(X_train, y_train)
# GPU
cuml_grid = GridSearchCV(cuml_ridge, params, scoring="r2")
cuml_grid.fit(X_train, y_train)
cuml_grid.best_params_, cuml_grid.best_score_
cuml_ridge = cuRidge(alpha = grid.best_params_["alpha"], fit_intercept=fit_intercept, solver="auto")
cuml_ridge.fit(X_train, y_train)
```

### 27. Performance comparison 1
```py
from cuml.benchmark.runners import SpeedupComparisonRunner
from cuml.benchmark.algorithms import algorithm_by_name
import pandas as pd
num_reps = 3
dataset_neighborhoods = "blobs"
dataset_classification = "classification"
dataset_regression = "regression"
input_type = "numpy"
benchmark_results = []
row_sizes = [2**x for x in range(14,17)]
features = [32,256]
def dic_result(algorithm, runner, result):
  result["algo"] = algorithm
  result["dataset_name"] = runner.dataset_name
  result["input_type"] = runner.input_type
  return result
def benchmark(algorithm, runner, verbose=True, run_cpu=True, **kwargs):
  results = runner.run(algorithm_by_name(algorithm), verbose=verbose, run_cpu=run_cpu, **kwargs)
  results = [dic_result(algorithm,runner,result) for result in results ]
  benchmark_results.extend(results)
```

### 28. Performance comparison 2
```py
runner = SpeedupComparisonRunner(bench_rows=row_sizes,
                                 bench_dims=features,
                                 dataset_name=dataset_regression,
                                 input_type=input_type,
                                 n_reps=num_reps)
benchmark("LinearRegression", runner) # took 6.3sec
'''
LinearRegression (n_samples=16384, n_features=32) [cpu=0.004033803939819336, gpu=0.0023708343505859375, speedup=1.701427996781979]
LinearRegression (n_samples=16384, n_features=256) [cpu=0.11832499504089355, gpu=0.019636154174804688, speedup=6.025874210781933]
LinearRegression (n_samples=32768, n_features=32) [cpu=0.006699800491333008, gpu=0.004670143127441406, speedup=1.4346028180518684]
LinearRegression (n_samples=32768, n_features=256) [cpu=0.33638858795166016, gpu=0.03228306770324707, speedup=10.419969720468226]
LinearRegression (n_samples=65536, n_features=32) [cpu=0.02423262596130371, gpu=0.009170055389404297, speedup=2.6425822890125317]
LinearRegression (n_samples=65536, n_features=256) [cpu=0.635448694229126, gpu=0.05872225761413574, speedup=10.821257901980925]
'''
# KMeans
runner = SpeedupComparisonRunner(bench_rows=row_sizes,
                                 bench_dims=features,
                                 dataset_name=dataset_neighborhoods,
                                 input_type=input_type,
                                 n_reps=num_reps)
benchmark("KMeans",runner)
'''
KMeans (n_samples=16384, n_features=32) [cpu=0.046213626861572266, gpu=0.03551125526428223, speedup=1.3013797039175534]
KMeans (n_samples=16384, n_features=256) [cpu=0.4621424674987793, gpu=0.06489324569702148, speedup=7.121580413105937]
KMeans (n_samples=32768, n_features=32) [cpu=0.06398606300354004, gpu=0.06985235214233398, speedup=0.9160187315261689]
KMeans (n_samples=32768, n_features=256) [cpu=0.5497612953186035, gpu=0.18251419067382812, speedup=3.012156442679486]
KMeans (n_samples=65536, n_features=32) [cpu=0.08124470710754395, gpu=0.11710405349731445, speedup=0.693782193537879]
KMeans (n_samples=65536, n_features=256) [cpu=0.907311201095581, gpu=0.4796757698059082, speedup=1.891509344869989]
'''
# Random Forest
runner = SpeedupComparisonRunner(bench_rows=row_sizes,
                                 bench_dims=features,
                                 dataset_name=dataset_classification,
                                 input_type=input_type,
                                 n_reps=num_reps)
benchmark("RandomForestClassifier",runner)
'''
RandomForestClassifier (n_samples=16384, n_features=32) [cpu=1.5299265384674072, gpu=0.2478952407836914, speedup=6.17166563436525]
RandomForestClassifier (n_samples=16384, n_features=256) [cpu=6.8781633377075195, gpu=0.6045911312103271, speedup=11.376553479933072]
RandomForestClassifier (n_samples=32768, n_features=32) [cpu=3.3415884971618652, gpu=0.3876461982727051, speedup=8.62020190589124]
RandomForestClassifier (n_samples=32768, n_features=256) [cpu=15.04035234451294, gpu=0.8101832866668701, speedup=18.564135538255318]
RandomForestClassifier (n_samples=65536, n_features=32) [cpu=8.894769191741943, gpu=0.8748669624328613, speedup=10.16699632479783]
RandomForestClassifier (n_samples=65536, n_features=256) [cpu=46.00590705871582, gpu=2.0520474910736084, speedup=22.419513806986036]
'''
```

### 29. Performance comparison 3

## Section 4: Complete project

### 30. Installations and libraries
```py
import numpy as np
import matplotlib.pyplot as plt
import cudf
import cupy as cp
```

### 31. Census dataset
```py
census = cudf.read_csv('census.csv')
census.isnull().sum()
```

### 32. Categorical features 1
```py
X_census = census.iloc[:, 0:14]
y_census = census.iloc[:,14] # income only
from cuml.preprocessing import LabelEncoder
label_encoder_test = LabelEncoder()
test = label_encoder_test.fit_transform(X_census["workclass"]) # label str into uint8
X_census["workclass"] = test
X_census.head()
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()
X_census['workclass'] = label_encoder_workclass.fit_transform(X_census['workclass'])
X_census['education'] = label_encoder_education.fit_transform(X_census['education'])
X_census['marital-status'] = label_encoder_marital.fit_transform(X_census['marital-status'])
X_census['occupation'] = label_encoder_occupation.fit_transform(X_census['occupation'])
X_census['relationship'] = label_encoder_relationship.fit_transform(X_census['relationship'])
X_census['race'] = label_encoder_race.fit_transform(X_census['race'])
X_census['sex'] = label_encoder_sex.fit_transform(X_census['sex'])
X_census['inative-country'] = label_encoder_country.fit_transform(X_census['inative-country'])
X_census = cp.from_dlpack(X_census.to_dlpack())
```

### 33. Categorical features 2
```py
from cuml.preprocessing import OneHotEncoder
from cuml.compose import ColumnTransformer
np.unique(census['workclass'])
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(sparse_output=False),
                                                        [1,3,4,6,7,8,9,13])], remainder='passthrough')
X_census = onehotencoder_census.fit_transform(X_census)
```

### 34. Additional pre-processing
```py
from cuml.preprocessing import StandardScaler
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)
from cuml.model_selection import train_test_split
y_census = LabelEncoder().fit_transform(y_census)
from cuml.model_selection import train_test_split
y_census = LabelEncoder().fit_transform(y_census)
X_census_train, X_census_test, y_census_train, y_census_test = train_test_split(X_census, y_census, test_size=0.15, random_state=42)
import pickle
with open('census.pkl', mode="wb") as f:
  pickle.dump([X_census_train, y_census_train, X_census_test, y_census_test],f)
```

### 35. Logistic regression and kNN
```py
with open('census.pkl', mode="rb") as f:
  X_census_train, y_census_train, X_census_test, y_census_test = pickle.load(f)
from cuml.linear_model import LogisticRegression
logistic_census = LogisticRegression()
logistic_census.fit(X_census_train, y_census_train)
predictions = logistic_census.predict(X_census_test)
from cuml.metrics import accuracy_score, confusion_matrix
accuracy_score(y_census_test, predictions) # 0.847
confusion_matrix(y_census_test, predictions, convert_dtype=True)
# array([[3428,  231],
#       [ 515,  710]])
from cuml.neighbors import KNeighborsClassifier
knn_census = KNeighborsClassifier(n_neighbors = 10)
knn_census.fit(X_census_train, y_census_train)
predictions = knn_census.predict(X_census_test)
accuracy_score(y_census_test, predictions) # 0.817
```

### 36. Random Forest and SVM
```py
from cuml.ensemble import RandomForestClassifier
random_forest_census = RandomForestClassifier(n_estimators=10,split_criterion="entropy", random_state=42)
random_forest_census.fit(X_census_train, y_census_train)
predictions = random_forest_census.predict(X_census_test)
accuracy_score(y_census_test, predictions) # 0.847
#
from cuml.svm import SVC
svm_census = SVC(kernel="linear", random_state=1)
svm_census.fit(X_census_train, y_census_train)
predictions = svm_census.predict(X_census_test)
accuracy_score(y_census_test, predictions) # 0.845
```

### 37. HOMEWORK

### 38. Homework solution 1

### 39. Homework solution 2

## Section 5: DASK

### 40. DASK - intuition
- Dask is a flexible open-source library for parallel computing in Python 
- For distributed processing
- Works with CPUs as well
- Distribute works by creating a Directed Acyclic Graph (DAG) representation of your code at runtime

### 41. Creating a local cluster
```py
import cudf, cuml
import dask, dask_cudf
print('Dask: ', dask.__version__)
print('Dask cuDF:', dask_cudf.__version__)
from dask.distributed import Client, LocalCluster, wait, progress
from dask_cuda import LocalCUDACluster
cluster = LocalCUDACluster(threads_per_worker=1)
client = Client(cluster)
client
```

### 42. Arrays in distributed GPUs
```py
import dask.array as da
import cupy as cp
rs = da.random.RandomState(RandomState=cp.random.RandomState, seed = 42)
x = rs.random((100000, 1000), chunks = (10000, 1000)) # (100k,1k)a matrix and partitioned as (10k,1k)
x = x.persist()
u, s, v = da.linalg.svd(x)
u, s, v = dask.persist(u, s, v)
```

### 43. DASK and cuDF
```py
s = cudf.Series([1,2,None,3,4])
ds = dask_cudf.from_cudf(s, npartitions=2)
ds.compute()
df = cudf.DataFrame(
    {'a': list(range(10)),
     'b': list(reversed(range(10))),
     'c': list(range(10))})
ddf = dask_cudf.from_cudf(df, npartitions=2)
ddf.compute()
ddf.sort_values(by = "b").compute()
ddf.query("b == 9").compute()
```

### 44. DASK and cuML 1
```py
from sklearn.metrics import accuracy_score
from sklearn import model_selection, datasets
from cuml.dask.common import utils as dask_utils
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf
from cuml.dask.ensemble import RandomForestClassifier as RF_cumlDask
from sklearn.ensemble import RandomForestClassifier as RF_skl
import pandas as pd
import numpy as np
cluster = LocalCUDACluster(threads_per_worker=1)
c = Client(cluster)
workers = c.has_what().keys()
workers # dict_keys(['tcp://127.0.0.1:33145'])
#
train_size = 100000
test_size = 1000
n_samples = train_size + test_size
n_features = 20
max_depth = 12
n_bins = 16
n_trees = 1000
# https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
X, y = datasets.make_classification(n_samples = n_samples, n_features = n_features,
                                    random_state = 42, n_classes = 5,
                                    n_informative = int(n_features / 3))
X = X.astype(np.float32)
y = y.astype(np.int32)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = test_size)
# distribution of data
n_partitions = n_workers
def distribute(X, y):
  X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X))
  y_cudf = cudf.Series(y)
  X_dask = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)
  y_dask = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)
  X_dask, y_dask = dask_utils.persist_across_workers(c, [X_dask, y_dask], workers=workers)
  return X_dask, y_dask
X_train_dask, y_train_dask = distribute(X_train, y_train)
X_test_dask, y_test_dask = distribute(X_test, y_test)
```

### 45. DASK and cuML 2
- CPU using scikit-learn
  - Using backend ThreadingBackend
```py
type(X_train), type(y_train)
skl_model = RF_skl(max_depth=max_depth, n_estimators=n_trees, n_jobs=-1, verbose=True)
skl_model.fit(X_train, y_train)
#[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
#[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    6.2s
#[Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   29.8s
#[Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  1.1min
#[Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:  1.8min
#[Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:  2.1min finished
```
- cuML + Dask
```py
type(X_train_dask), type(y_train_dask)
n_streams = 8
cuml_model = RF_cumlDask(max_depth=max_depth, n_estimators=n_trees, n_bins=n_bins, n_streams=n_streams)
cuml_model.fit(X_train_dask, y_train_dask)
wait(cuml_model.rfs)
cuml_model.predict(X_test_dask).compute()
```

## Section 6: Final remarks

### 46. Final remarks

### 47. BONUS


