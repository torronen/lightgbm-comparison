# lightgbm-comparison
Compare LightGBM Python and .NET interfaces

To run Python script, use LightGBM version 2.3.1 to match ML.NET

Results for Titanic dataset (train.csv for training, test.csv for validation)
- FLAML accuracy 98+% 
- LightGBM in ML.NET accuracy 90% 
- ModelBuilder Binary:FastTree 98+%, LightGBM 92% (reported by Model Builder which creates it's own test data from train.csv)
- Multiclass model builder has slightly slower scores, see .mbconfig files.

Results should also be compared with feature_fraction and other parameters which use randomity.

Suggested changes (not reflected in results yet)
https://github.com/dotnet/machinelearning/pull/6064
