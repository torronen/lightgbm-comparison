﻿// This file was auto-generated by ML.NET Model Builder. 
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
namespace LightGbmTrainer
{
    public partial class Titanic_binary
    {
        /// <summary>
        /// model input class for Titanic_binary.
        /// </summary>
        #region model input class
        public class ModelInput
        {
            [ColumnName(@"survived")]
            public bool Survived { get; set; }

            [ColumnName(@"pclass")]
            public float Pclass { get; set; }

            [ColumnName(@"sex")]
            public string Sex { get; set; }

            [ColumnName(@"age")]
            public float Age { get; set; }

            [ColumnName(@"sibsp")]
            public float Sibsp { get; set; }

            [ColumnName(@"parch")]
            public float Parch { get; set; }

            [ColumnName(@"ticket")]
            public string Ticket { get; set; }

            [ColumnName(@"fare")]
            public float Fare { get; set; }

            [ColumnName(@"cabin")]
            public string Cabin { get; set; }

            [ColumnName(@"embarked")]
            public string Embarked { get; set; }

            [ColumnName(@"boat")]
            public string Boat { get; set; }

            [ColumnName(@"body")]
            public float Body { get; set; }

            [ColumnName(@"home.dest")]
            public string Home_dest { get; set; }

            [ColumnName(@"surname")]
            public string Surname { get; set; }

            [ColumnName(@"title")]
            public string Title { get; set; }

            [ColumnName(@"firstname")]
            public string Firstname { get; set; }

            [ColumnName(@"random")]
            public float Random { get; set; }

        }

        #endregion

        /// <summary>
        /// model output class for Titanic_binary.
        /// </summary>
        #region model output class
        public class ModelOutput
        {
            [ColumnName(@"survived")]
            public bool Survived { get; set; }

            [ColumnName(@"pclass")]
            public float Pclass { get; set; }

            [ColumnName(@"sex")]
            public float[] Sex { get; set; }

            [ColumnName(@"age")]
            public float Age { get; set; }

            [ColumnName(@"sibsp")]
            public float Sibsp { get; set; }

            [ColumnName(@"parch")]
            public float Parch { get; set; }

            [ColumnName(@"ticket")]
            public float[] Ticket { get; set; }

            [ColumnName(@"fare")]
            public float Fare { get; set; }

            [ColumnName(@"cabin")]
            public float[] Cabin { get; set; }

            [ColumnName(@"embarked")]
            public float[] Embarked { get; set; }

            [ColumnName(@"boat")]
            public float[] Boat { get; set; }

            [ColumnName(@"body")]
            public float Body { get; set; }

            [ColumnName(@"home.dest")]
            public float[] Home_dest { get; set; }

            [ColumnName(@"surname")]
            public float[] Surname { get; set; }

            [ColumnName(@"title")]
            public float[] Title { get; set; }

            [ColumnName(@"firstname")]
            public float[] Firstname { get; set; }

            [ColumnName(@"random")]
            public float Random { get; set; }

            [ColumnName(@"Features")]
            public float[] Features { get; set; }

            [ColumnName(@"PredictedLabel")]
            public bool PredictedLabel { get; set; }

            [ColumnName(@"Score")]
            public float Score { get; set; }

            [ColumnName(@"Probability")]
            public float Probability { get; set; }

        }

        #endregion

        private static string MLNetModelPath = Path.GetFullPath("titanic-binary.zip");

        public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);

        /// <summary>
        /// Use this method to predict on <see cref="ModelInput"/>.
        /// </summary>
        /// <param name="input">model input.</param>
        /// <returns><seealso cref=" ModelOutput"/></returns>
        public static ModelOutput Predict(ModelInput input)
        {
            var predEngine = PredictEngine.Value;
            return predEngine.Predict(input);
        }

        private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }
    }
}