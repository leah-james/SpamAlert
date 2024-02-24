
using Microsoft.ML.Data;

namespace SpamAlert
{
    internal class ModelInput
    {

        [LoadColumn(0)]
        [ColumnName("Label")]
        public string? Label { get; set; }

        [LoadColumn(1)]
        [ColumnName("Text")]
        public string? Text { get; set; }
    }
}
