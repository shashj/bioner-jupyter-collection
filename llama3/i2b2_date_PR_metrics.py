import pickle
import re




with open("../datasets/i2b2/train_jsons/all_records_train_text.pkl", 'rb') as f:
    ground_truth_records_text = pickle.load(f)

with open("results_dates/train_date_results.pkl", 'rb') as f:
    dates_results = pickle.load(f)

with open("../datasets/i2b2/train_jsons/all_records_train.pkl", 'rb') as f:
    ground_truth_records = pickle.load(f)


def find_match_offsets(text, pattern):
    escaped_pattern = re.escape(pattern)
    matches = re.finditer(escaped_pattern, text)
    return [(match.start(), match.end()) for match in matches]

def calculate_precision_recall(ground_truth, predicted_text):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Create dictionaries mapping offsets to their corresponding text
    ground_truth_dict = {}
    for item in ground_truth.values():
        for offset in item['offsets']:
            ground_truth_dict[offset] = item['ground_truth']

    predicted_dict = {}
    for item in predicted_text.values():
        for offset in item['offsets']:
            predicted_dict[offset] = item['predicted_text']

    # Calculate true positives and false positives
    for pred_offset, pred_text in predicted_dict.items():
        pred_start, pred_end = pred_offset
        match_found = False
        for gt_offset, gt_text in ground_truth_dict.items():
            gt_start, gt_end = gt_offset
            if (pred_start <= gt_end and gt_start <= pred_end):
                # Check if the predicted text contains the ground truth text
                if (gt_text in pred_text) | (pred_text in gt_text):
                    true_positives += 1
                    match_found = True
                    break
        if not match_found:
            false_positives += 1

    # Calculate false negatives
    for gt_offset, gt_text in ground_truth_dict.items():
        gt_start, gt_end = gt_offset
        match_found = False
        for pred_offset, pred_text in predicted_dict.items():
            pred_start, pred_end = pred_offset
            if (pred_start <= gt_end and gt_start <= pred_end):
                if (gt_text in pred_text) | (pred_text in gt_text):
                    match_found = True
                    break
        if not match_found:
            false_negatives += 1

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall, true_positives, false_positives, false_negatives

def get_precision_recall_for_all_records(ground_truth_records_text, dates_results, ground_truth_records):
    results = {}
    records = dates_results.keys()
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    filtered_dict = {key: [item['TEXT'] for item in value if item['TYPE'] == 'DATE'] for key, value in ground_truth_records.items()}

    for record in records:

        print(f"PR for record: {record}")
        if list(dates_results[record].dict().values())[0] is None:
            continue

        # get predicted results
        include_vector = [
            (re.search(rf'(\(\s*{x}\s*\)|\b{x}\b)', ground_truth_records_text[record]).group()
            if re.search(rf'(\(\s*{x}\s*\)|\b{x}\b)', ground_truth_records_text[record])
            else None)
            if x[0].isdigit() and int(x[0])!=0 
            else (
                (re.search(rf'(\(\s*{x[1:]}\s*\)|\b{x[1:]}\b)', ground_truth_records_text[record]).group()
                if re.search(rf'(\(\s*{x[1:]}\s*\)|\b{x[1:]}\b)', ground_truth_records_text[record])
                else None) or
                (re.search(rf'(\(\s*{x}\s*\)|\b{x}\b)', ground_truth_records_text[record]).group()
                if re.search(rf'(\(\s*{x}\s*\)|\b{x}\b)', ground_truth_records_text[record])
                else None)
            )
            for x in list(dates_results[record].dict().values())[0]
        ]

        # Remove None values
        final_predictions = [item for item in include_vector if item is not None]

        ## offsets for predicted

        results_pred = {
            item: {
                'offsets': offsets,
                'predicted_text': item
            }
            for item in final_predictions
            if (offsets := find_match_offsets(ground_truth_records_text[record], item))
        }

        ## offsets for GT

        results_gt = {
            item: {
                'offsets': offsets,
                'ground_truth': item
            }
            for item in filtered_dict[record]
            if (offsets := find_match_offsets(ground_truth_records_text[record], item))
        }

        # Calculate precision and recall
        precision, recall, true_positives, false_positives, false_negatives = calculate_precision_recall(results_gt, results_pred)
        # Add to totals
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        
        results[record] = {
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

        print(f"Record: {record}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

    # Calculate overall precision and recall
    overall_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    overall_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    
    # Add overall results to the results dictionary
    results['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'true_positives': total_true_positives,
        'false_positives': total_false_positives,
        'false_negatives': total_false_negatives
    }
    print("Overall:")
    print(f"Overall Precision: {overall_precision:.2f}")
    print(f"Overall Recall: {overall_recall:.2f}")
        
    return results


if __name__=="__main__":

    results = get_precision_recall_for_all_records(ground_truth_records_text, dates_results, ground_truth_records)

    # Save results to a pickle file
    with open('dates_precision_recall_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Results have been saved to 'dates_precision_recall_results.pkl'")

