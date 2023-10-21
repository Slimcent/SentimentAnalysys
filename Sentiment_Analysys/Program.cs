using Microsoft.ML;
using Microsoft.ML.Data;
using Sentiment_Analysis;
using Sentiment_Analysis.Models;
using System;
using static Microsoft.ML.DataOperationsCatalog;

MLContext mlContext = new();

// Load and split the data
TrainTestData splitDataView = LoadAndSplitData.LoadData(mlContext);

// Build and train the model
ITransformer model = BuildingAndTrainingModel.BuildAndTrainModel(mlContext, splitDataView.TrainSet);

// Displaying the metrics for model validation
Evaluation.Evaluate(mlContext, model, splitDataView.TestSet);

// Predict the test data outcome on a single item
ModelPrediction.UseModelWithSingleItem(mlContext, model);

// Predict the test data outcome on many items
ModelPrediction.UseModelWithBatchItems(mlContext, model);


