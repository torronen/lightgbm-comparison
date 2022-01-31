# lightgbm-comparison
Compare LightGBM Python and .NET interfaces

To run Python script, use LightGBM version 2.3.1 to match ML.NET

Results for Titanic dataset (train.csv for training, test.csv for validation)
- FLAML accuracy 98+% 
- LightGBM in ML.NET accuracy 90% with params from FLAML
- ModelBuilder Binary:FastTree 98+%, LightGBM 92% (reported by Model Builder which creates it's own test data from train.csv)
- Multiclass model builder has slightly slower scores, see .mbconfig files.

Results should also be compared with feature_fraction and other parameters which use randomity. Titanic dataset did not benefit from them according the tuning process. We probably can still try feature_fraction on Titanic dataset and compare results, although it may be better to also test with datasets that will benefit from it.

Suggested changes (not reflected in results yet)
https://github.com/dotnet/machinelearning/pull/6064
