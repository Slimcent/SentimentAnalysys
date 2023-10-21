using Microsoft.ML;
using Sentiment_Analysis.Models;
using static Microsoft.ML.DataOperationsCatalog;

namespace Sentiment_Analysis
{
    public static class LoadAndSplitData
    {
        public static TrainTestData LoadData(MLContext mlContext)
        {
            string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }
    }
}
