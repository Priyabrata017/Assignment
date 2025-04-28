from transformers import EvalPrediction
from collections import defaultdict

def compute_metrics(p: EvalPrediction):
    predictions, labels = p.predictions, p.label_ids  # 'labels' might need different handling

    # 1. Parse Model Predictions into Spans
    predicted_spans_batch = []
    for i, output in enumerate(predictions):
        predicted_spans = get_predicted_spans(output, p.inputs[i]) # You need to implement this function
        predicted_spans_batch.append(predicted_spans)

    # 2. Format Ground Truth Spans
    ground_truth_spans_batch = []
    for i, label_sequence in enumerate(labels):
        ground_truth_spans = get_ground_truth_spans(label_sequence, p.inputs[i]) # You need to implement this function
        ground_truth_spans_batch.append(ground_truth_spans)

    # 3. Compare Spans and Calculate Metrics
    results = compute_span_metrics(predicted_spans_batch, ground_truth_spans_batch) # You need to implement this function
    return results

def get_predicted_spans(model_output, input_sequence):
    """
    Parses the model's output to extract predicted entity spans.
    This function needs to be implemented based on your GLiNER model's specific output format.
    It should return a list of tuples: (start_index, end_index, entity_type).
    """
    # Placeholder - Implement based on your model's output
    predicted_spans = []
    # Example: If your model outputs a list of (start, end, type)
    # for prediction in model_output:
    #     start, end, type = ...
    #     predicted_spans.append((start, end, type))
    return predicted_spans

def get_ground_truth_spans(label_sequence, input_sequence):
    """
    Formats the ground truth labels into a list of entity spans.
    This function needs to be implemented based on the format of your evaluation labels.
    It should return a list of tuples: (start_index, end_index, entity_type).
    """
    # Placeholder - Implement based on your label format
    ground_truth_spans = []
    # Example: If your labels are IOB or similar, you'll need to process them
    # to identify spans and their types.
    return ground_truth_spans

def compute_span_metrics(predicted_spans_batch, ground_truth_spans_batch):
    """
    Compares the predicted spans with the ground truth spans and calculates precision, recall, and F1-score.
    """
    all_predicted_spans = [span for seq_spans in predicted_spans_batch for span in seq_spans]
    all_ground_truth_spans = [span for seq_spans in ground_truth_spans_batch for span in seq_spans]

    tp = 0
    fp = 0
    fn = 0

    for pred_span in all_predicted_spans:
        if pred_span in all_ground_truth_spans:
            tp += 1
        else:
            fp += 1

    for gt_span in all_ground_truth_spans:
        if gt_span not in all_predicted_spans:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    return results
