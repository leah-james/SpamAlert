using System.Collections.Generic;
using System.Formats.Asn1;
using System.Globalization;
using System.IO;
using CsvHelper;
using Microsoft.ML;
using SpamAlert;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;

List<ModelInput> spamData = ReadSpamFile("spam.csv");

MLContext mlContext = new MLContext();
mlContext.GpuDeviceId = 0;
mlContext.FallbackToCpu = true;

var reviewsDV = mlContext.Data.LoadFromEnumerable(spamData);

var pipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label")
    .Append(mlContext.MulticlassClassification.Trainers.TextClassification(sentence1ColumnName: "Text", labelColumnName: "LabelKey", architecture: BertArchitecture.Roberta))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

var model = pipeline.Fit(reviewsDV);

PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

while (true)
{
    Console.WriteLine("Enter a text message:");
    string userText = Console.ReadLine();
    if (userText == "quit")
    {
        break;
    }
    ModelInput modelInput = new ModelInput();
    modelInput.Text = userText;
    ModelOutput result = engine.Predict(modelInput);
    Console.WriteLine($"Predicted Label: {result.PredictedLabel}");
}

List<ModelInput> ReadSpamFile(string filename)
{
    using (var reader = new StreamReader(Path.Combine(Directory.GetCurrentDirectory(), filename)))
    using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
    {
        var records = csv.GetRecords<ModelInput>();
        return records.ToList();
    }
}
