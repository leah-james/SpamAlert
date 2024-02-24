using Microsoft.ML.Data;

namespace SpamAlert
{
    internal class ModelOutput
    {
        [ColumnName("Text")]
        public string Text { get; set; }

        [ColumnName("Label")]
        public string Label { get; set; }

        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName("Score")]
        public float[] Score { get; set; }
    }
}
