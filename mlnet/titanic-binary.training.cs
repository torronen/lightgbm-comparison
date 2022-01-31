﻿﻿// This file was auto-generated by ML.NET Model Builder. 
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers;
using Microsoft.ML;

namespace LightGbmTrainer
{
    public partial class Titanic_binary
    {
        public static ITransformer RetrainPipeline(MLContext context, IDataView trainData)
        {
            var pipeline = BuildPipeline(context);
            var model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new []{new InputOutputColumnPair(@"sex", @"sex"),new InputOutputColumnPair(@"ticket", @"ticket"),new InputOutputColumnPair(@"cabin", @"cabin"),new InputOutputColumnPair(@"embarked", @"embarked"),new InputOutputColumnPair(@"boat", @"boat"),new InputOutputColumnPair(@"home.dest", @"home.dest"),new InputOutputColumnPair(@"surname", @"surname"),new InputOutputColumnPair(@"title", @"title"),new InputOutputColumnPair(@"firstname", @"firstname")})      
                                    .Append(mlContext.Transforms.ReplaceMissingValues(new []{new InputOutputColumnPair(@"pclass", @"pclass"),new InputOutputColumnPair(@"age", @"age"),new InputOutputColumnPair(@"sibsp", @"sibsp"),new InputOutputColumnPair(@"parch", @"parch"),new InputOutputColumnPair(@"fare", @"fare"),new InputOutputColumnPair(@"body", @"body"),new InputOutputColumnPair(@"random", @"random")}))      
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"sex",@"ticket",@"cabin",@"embarked",@"boat",@"home.dest",@"surname",@"title",@"firstname",@"pclass",@"age",@"sibsp",@"parch",@"fare",@"body",@"random"}))      
                                    .Append(mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options(){NumberOfLeaves=12,MinimumExampleCountPerLeaf=2,NumberOfTrees=31,MaximumBinCountPerFeature=218,FeatureFraction=0.973461065469314,LearningRate=0.141097909168156,LabelColumnName=@"survived",FeatureColumnName=@"Features"}))      
                                    .Append(mlContext.BinaryClassification.Calibrators.Naive(labelColumnName:@"survived",scoreColumnName:@"Score"));

            return pipeline;
        }
    }
}
