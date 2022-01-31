using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace Kwork
{
    internal static class LightGBMTrainer
    {
        // NOTE: Titanic dataset can be used only to compare simple cases
        // Dataset with more columns might be needed to test feature_fraction and others

        // LightGBM hyperparameters from python
        // get values from flaml-best-models.txt log
        
        // FLAML reports 98.3% accuracy with these parameters
        static Single learning_rate = 0.09999999999999995f;
        static int max_bin = 255;
        static int n_estimators = 4;
        static int num_leaves = 4;
        static Single reg_alpha = 0.0009765625f;
        static Single reg_lambda = 1.0f;

        // other datasets may tune these values also:
        //static Single colsample_bytree = 0.9040351012809469f;
        //static int min_child_samples = 19;


        // if you only have log_max_bin from logs, uncomment this:
        ///max_bin = (1 << log_max_bin) - 1;

        static string traindatapath = "train.csv";
        static string validationdatapath = "test.csv";
        static string labelColumnName = "survived";


        public static void Train()
        {
                var mlContext = new MLContext(seed: 123);

                // read data 
                ColumnInferenceResults columnInference = GetColumnInference(traindatapath, labelColumnName, mlContext);
                columnInference.TextLoaderOptions.UseThreads = true;
                columnInference.ColumnInformation.LabelColumnName = labelColumnName;

                var trainingData = LoadData(traindatapath, mlContext, columnInference);
               
                // Define trainer options.
                var options = new LightGbmBinaryTrainer.Options
                {
                    LabelColumnName = labelColumnName,
                    LearningRate = learning_rate,
                    MaximumBinCountPerFeature = max_bin,
                    //MinimumExampleCountPerLeaf = min_child_samples,
                    NumberOfIterations = n_estimators,
                    NumberOfLeaves = num_leaves,

                    Seed = 123,
                    UseCategoricalSplit = true,
                    UseZeroAsMissingValue = true,

                    Booster = new GradientBooster.Options
                    {
                        L1Regularization = reg_alpha,
                        L2Regularization = reg_lambda,
                        //FeatureFraction = colsample_bytree,

                        // defaults https://github.com/microsoft/LightGBM/blob/v2.3.1/include/LightGBM/config.h
                        SubsampleFraction = 1.0, // bagging_fraction  
                        SubsampleFrequency = 0, //bagging_freq 
                                                //TopRate = 0.2, // top_rate 
                                                //OtherRate = 0.1, // other_rate 
                        MaximumTreeDepth = -1, // max_depth
                                               // MinimumChildWeight = unknown;
                        MinimumSplitGain = 0.0, // min_gain_to_split 
                        MinimumChildWeight = 1e-3 // https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
                    },

                    Verbose = true,
                    Silent = false,

                    // defaults https://github.com/microsoft/LightGBM/blob/v2.3.1/include/LightGBM/config.h
                    MinimumExampleCountPerGroup = 100, // min_data_per_group 
                    MaximumCategoricalSplitPointCount = 32, // max_cat_threshold
                    CategoricalSmoothing = 10.0, // cat_smooth 
                    //NumberOfThreads = 0,
                    HandleMissingValue = true, //use_missing
                    UnbalancedSets = false, //is_unbalance
                    Sigmoid = 1, //sigmoid
                    L2CategoricalRegularization = 10, // cat_l2
                                                      // UseCategoricalSplit = unknown

                };


                // Train the model.

                var trainer = mlContext.BinaryClassification.Trainers
                  .LightGbm(options);

                IEstimator<ITransformer> pipeline = BuildPipeline(mlContext, trainer);

                Console.WriteLine("start fitting");
                var model = pipeline.Fit(trainingData); //trainingData.ToDataFrame(-1)

                // Read test data
                Console.WriteLine("read test data");
                var testData = LoadData(validationdatapath, mlContext, columnInference);

                // Run the model on test data set.
                var transformedTestData = model.Transform(testData);

                // Evaluate the overall metrics.
                var metrics = mlContext.BinaryClassification
                    .Evaluate(transformedTestData, labelColumnName);

                PrintMetrics(metrics);
        }

        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext, IEstimator<ITransformer> trainer)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair(@"sex", @"sex"), new InputOutputColumnPair(@"ticket", @"ticket"), new InputOutputColumnPair(@"cabin", @"cabin"), new InputOutputColumnPair(@"embarked", @"embarked"), new InputOutputColumnPair(@"boat", @"boat"), new InputOutputColumnPair(@"home.dest", @"home.dest"), new InputOutputColumnPair(@"surname", @"surname"), new InputOutputColumnPair(@"title", @"title"), new InputOutputColumnPair(@"firstname", @"firstname") })
                                    .Append(mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"pclass", @"pclass"), new InputOutputColumnPair(@"age", @"age"), new InputOutputColumnPair(@"sibsp", @"sibsp"), new InputOutputColumnPair(@"parch", @"parch"), new InputOutputColumnPair(@"fare", @"fare"), new InputOutputColumnPair(@"body", @"body"), new InputOutputColumnPair(@"random", @"random") }))
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"sex", @"ticket", @"cabin", @"embarked", @"boat", @"home.dest", @"surname", @"title", @"firstname", @"pclass", @"age", @"sibsp", @"parch", @"fare", @"body", @"random" }))
                                    .Append(trainer);
                                    //.Append(mlContext.BinaryClassification.Calibrators.Naive(labelColumnName: @"survived", scoreColumnName: @"Score"));

            return pipeline;
        }

        #region Helpers
        public static IDataView LoadData(string csvPath, MLContext mlContext, ColumnInferenceResults columnInference)
        {
            var loadOptions = columnInference.TextLoaderOptions;
            loadOptions.UseThreads = false;
            loadOptions.AllowQuoting = false;
            loadOptions.TrimWhitespace = true;
            loadOptions.ReadMultilines = false;
            loadOptions.AllowSparse = false;
            loadOptions.DecimalMarker = '.';
            loadOptions.HasHeader = true;
            loadOptions.MissingRealsAsNaNs = false;
            loadOptions.Separators= new char[] { ',' };

            TextLoader textLoader = mlContext.Data.CreateTextLoader(
                options: loadOptions);
            var dataview = textLoader.Load(csvPath);
            return dataview;
        }

        public static ColumnInferenceResults GetColumnInference(string predictionDatasetPath, string labelColumnName, MLContext mlContext)
        {
            ColumnInferenceResults columnInference;
            columnInference = mlContext.Auto().InferColumns(predictionDatasetPath, labelColumnName, groupColumns: false);
            return columnInference;
        }

        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
  
        }
        #endregion

    }
}
