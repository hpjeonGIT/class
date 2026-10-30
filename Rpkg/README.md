# Make a custom package for R
- Ref: 
  - https://bioconductor.org/developers/how-to/buildingPackagesForBioc/
  - https://blog.albertkuo.me/post/creating-an-r-package-for-the-first-time/
  - https://github.com/duffau/my_first_r_package
  - https://www.lancaster.ac.uk/stor-i-student-sites/ziyang-yang/2021/03/13/3-steps-to-build-own-r-package-rcpp/
  
## Pre-requisites
- sudo apt install libxml2-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libwebp-dev libuv1-dev 
- In R, `install.packages(c("devtools","roxygen2"))`

## Project 1
- A function returns x+1 for a given input x
```
> library(devtools)
> library(roxygen2)
> setwd(~/hw/class/Rpkg/new_project/)
> create("MyPkg")
```
- File/Folder structure:
```bash
 MyPkg/
 ├── DESCRIPTION
 ├── NAMESPACE
 └── R
     └── MyPkg.R 
```
- DESCRIPTION
```txt
Package: MyPkg
Title: A function returns x+1
Version: 0.0.0.9000
Authors@R: 
    person("Byoungseon", "Jeon", email="bjeon11@gmail.com", role = c("aut","cre"))
Description: A test package for R/BioConductor. 
License: `use_gpl3_license()`
Config/roxygen2/version: 8.1.1
Encoding: UTF-8
Roxygen: list(markdown = TRUE)
```
- NAMESPACE
```txt
export('myFun')
import(Rcpp)
```
- R/MyPkg.R
```R
myFun <- function(x) { x +1 }
```
- Command to install: `R CMD INSTALL MyPkg`
- Demo:
```R
> library(MyPkg)
> myFun(1)
[1] 2
```

## Project 2
- R package with Rcpp
  - A Cpp function returns an array of two array sum
- Note: we run Rcpp skeleton() package to produce a package structure
    1. First, write a Cpp code with Rcpp grammar
    2. Using the Cpp code, Rcpp generates package folder/files
    3. The produced files/folders work as a single R package
- vectorSum.cpp:
```cpp
#include <Rcpp.h>
using namespace Rcpp;
//' @export
// [[Rcpp::export]]
NumericVector sum_arrays(NumericVector x, NumericVector y) {
  int n = x.size();
  NumericVector result(n);
  for(int i = 0; i < n; i++) {
    if (i%1000 == 0) {Rcpp:checkUserInterrupt();} // interrupt when the vector is too long
    result[i] = x[i] + y[i];
  }
  return result;
}
```
- Run Rcpp.package.skeleton() to produce folder structures and auto-generated wrapper files
```R
> library(Rcpp)
> Rcpp.package.skeleton("MyRcppPkg", cpp_files="./vectorSum.cpp", example_code=FALSE)
Creating directories ...
Creating DESCRIPTION ...
Creating NAMESPACE ...
Creating Read-and-delete-me ...
Saving functions and data ...
Making help files ...
Done.
Further steps are described in './MyRcppPkg/Read-and-delete-me'.

Adding Rcpp settings
 >> added Imports: Rcpp
 >> added LinkingTo: Rcpp
 >> added useDynLib directive to NAMESPACE
 >> added importFrom(Rcpp, evalCpp) directive to NAMESPACE
 >> copied ./vectorSum.cpp to src directory
 >> compiled Rcpp attributes 
```
- Produced file/folder structure:
```bash
MyRcppPkg/
├── DESCRIPTION
├── man
│   └── MyRcppPkg-package.Rd
├── NAMESPACE
├── R
│   └── RcppExports.R # this file is generated from Rcpp
├── Read-and-delete-me
└── src
    ├── RcppExports.cpp  # this file is generated from Rcpp
    └── vectorSum.cpp
```
- Install step command: `R CMD INSTALL ./MyRcppPkg`
- Demo:
```R
> library(MyRcppPkg)
> x = rnorm(10,2,4)
> y = rnorm(10,2,10)
> z <- sum_arrays(x,y)
```
- To remove the installed Package: `R CMD REMOVE MyRcppPkg`

## Project 3
- Implementing OpenMP for array sum
- Basically repeating Project 2 above
- vectorSum_omp.cpp
```cpp
#include <Rcpp.h>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;
//' @export
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]
NumericVector sum_arrays(NumericVector x, NumericVector y) {
  #ifdef _OPENMP
  #pragma omp parallel
  #pragma omp master
  std::cout << "Number of OpenMP threads = " << omp_get_num_threads() << std::endl;
  #endif
  int n = x.size();
  NumericVector result(n);
  #pragma omp parallel default(private) shared(x,y,result,n)
  {
    #pragma omp for schedule(static)
    for(int i = 0; i < n; i++) {
      if (i%1000 == 0) {Rcpp:checkUserInterrupt();} // interrupt when the vector is too long
      result[i] = x[i] + y[i];
    }
  }
  return result;
}
```
- Now run `Rcpp.package.skeleton("MyOMPPkg", cpp_files="./vectorSum_omp.cpp", example_code=FALSE)`
  - It will generate the similar files/folders structure above but we need extra files such as:
  - src/Makevars
```  
PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)
PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)
```
- Update DESCRIPTION/NAMESPACE as necessary
- Final folders/files structure:
```bash
MyOMPPkg/
├── DESCRIPTION
├── man
│   └── MyOMPPkg-package.Rd
├── NAMESPACE
├── R
│   └── RcppExports.R
├── Read-and-delete-me
└── src
    ├── Makevars
    ├── RcppExports.cpp
    └── vectorSum_omp.cpp
```
- Install step command: `R CMD INSTALL ./MyOMPPkg`
  - Make sure that openmp flag (like `-fopenmp` in gcc) appears when the source code is compiled/linked
- Demo:
```R
> library(MyOMPPkg)
> x = rnorm(10,2,4)
> y = rnorm(10,2,4)
> z <- sum_arrays(x,y)
Number of OpenMP threads = 2
```

## Project 4
- Enabling OpenMP in DESeq2 package
- Check `lib/R/etc/Makeconf` if OPENMP flag is configured
  - SHLIB_OPENMP_CXXFLAGS or SHLIB_OPENMP_FFLAGS or SHLIB_OPENMP_CFLAGS may exist
  - Edit src/Makevars as:
```
PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)
PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)
```    
  - Modification of DESeq2.cpp
```cpp
// user includes
#include <R.h>
#include <Rmath.h>
#include <R_ext/Utils.h>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
...
#ifdef _OPENMP
// [[Rcpp::plugins(openmp)]]
#endif
// Obtain the MLE or MAP dispersion estimate using line search.
// fitting occurs on the scale of log(alpha)
//
// [[Rcpp::export]]
List fitDisp(...){
  NumericMatrix y(ySEXP);
  arma::mat x = as<arma::mat>(xSEXP);
  int y_n = y.nrow();
  #ifdef _OPENMP
    #pragma omp parallel
    #pragma omp master
    std::cout << "In fitDisp(). Number of threads for openMP = " << omp_get_num_threads() << std::endl;
    std::cout << "y_n = " << y_n << std::endl;
  #endif
...
  bool useCR = as<bool>(useCRSEXP);
  #pragma omp parallel default(private) shared(x, y, y_n, log_alpha, mu_hat, log_alpha_prior_mean, log_alpha_prior_sigmasq, min_log_alpha, kappa_0, maxit, epsilon, initial_lp, initial_dlp, last_lp, last_dlp, last_d2lp, last_change, iter, iter_accept, tol, usePrior, weights, useWeights, weightThreshold, useCR)
  {
  #pragma omp for schedule(static)
  for (int i = 0; i < y_n; i++) { 
...  
List fitBeta(...) {
...
  int y_n = y.n_rows;
...
  #ifdef _OPENMP
    #pragma omp parallel
    #pragma omp master
    std::cout << "In fitBeta(). Number of threads for openMP = " << omp_get_num_threads() << std::endl;
    std::cout << "y_n = " << y_n << std::endl;
  #endif 
...
  #pragma omp parallel default(private) shared(y, nf, x, y_n, y_m, x_p, alpha_hat, beta_mat, beta_var_mat, contrast_num, contrast_denom, hat_diagonals, lambda, contrast, maxit, weights, useWeights, useQR, tol, minmu, large, iter, deviance)  
  {
  #pragma omp for schedule(static)
  for (int i = 0; i < y_n; i++) {
...    
```
- Demo:
```bash
$ ulimit -s unlimited # May need very large stack memory
$ Rscript serial.R    # Sample code/data from Udemy class
...
Starting DESeq()
estimating size factors
estimating dispersions
gene-wise dispersion estimates
mean-dispersion relationship
final dispersion estimates
fitting model and testing
Took 4.678947 sec
...
$ export OMP_NUM_THREADS=4
$ Rscript omp_run.R   # 
...
Starting DESeq()
estimating size factors
estimating dispersions
gene-wise dispersion estimates
In fitDisp(). Number of threads for openMP = 4
y_n = 22369
mean-dispersion relationship
final dispersion estimates
In fitDisp(). Number of threads for openMP = 4
y_n = 22369
fitting model and testing
In fitBeta(). Number of threads for openMP = 4
y_n = 22369
Took 3.80582 sec # <------------- Reduced but not much difference
...
```
- Confirmed that they yield the same results
- Performance gain is very slight
  - Too small data set? (22,369 might not large enough for parallel processing)
