# lightgbm-comparison
Compare LightGBM Python and .NET interfaces

To run Python script, use LightGBM version 2.3.1 to match ML.NET

Results
- FLAML accuracy 98+%
- LightGBM in ML.NET accuracy 90%
 - ModelBuilder Binary:FastTree 98+%, LightGBM 92%

Multiclass model builder has slightly slower scores, see .mbconfig files.

The results above, especially ModelBuilder's low scores for LightGBM, suggest there may be some misconfiguration in LightGBM .NET interface.
