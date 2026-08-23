import json
import argparse
import random
import nltk
from nltk.tokenize import sent_tokenize

# Ensure the NLTK sentence tokenizer is available.
nltk.download('punkt', quiet=True)

def load_jsonl(filepath):
    """Load a JSONL file and return a list of JSON objects."""
    data = []
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def find_added_sentences(orig_context, adv_context):
    """
    Tokenize both contexts into sentences and return a list of sentences
    that are present in the adversarial context but not in the original.
    """
    orig_sentences = set(sent_tokenize(orig_context))
    adv_sentences = sent_tokenize(adv_context)
    added = [sent for sent in adv_sentences if sent not in orig_sentences]
    return added

def normalize(text):
    """Normalize a string by lowercasing and stripping whitespace."""
    return text.lower().strip()

def main():
    parser = argparse.ArgumentParser(
        description="Compare two JSONL files (original/without and adversarial/with) and output examples where the predicted answers differ."
    )
    parser.add_argument("file1", help="Path to the original/without JSONL file (should contain 'predicted_answer' and 'answers')")
    parser.add_argument("file2", help="Path to the adversarial/with JSONL file (should contain 'predicted_answer')")
    parser.add_argument("-n", type=int, default=None,
                        help="Number of differing examples to output (if omitted, output all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for selecting samples (default: 42)")
    args = parser.parse_args()

    # Load both JSONL files.
    data_orig = load_jsonl(args.file1)
    data_adv = load_jsonl(args.file2)

    # Create dictionaries mapping id -> sample.
    orig_dict = {sample["id"]: sample for sample in data_orig}
    adv_dict = {sample["id"]: sample for sample in data_adv}

    differing_examples = []
    common_ids = set(orig_dict.keys()) & set(adv_dict.keys())
    for id_ in common_ids:
        sample_orig = orig_dict[id_]
        sample_adv = adv_dict[id_]
        # Pull predicted answers from each file.
        pred_orig = sample_orig.get("predicted_answer", "")
        pred_adv = sample_adv.get("predicted_answer", "")
        # Only keep examples where the normalized predicted answers differ.
        if normalize(pred_orig) == normalize(pred_adv):
            continue

        # Compute added context (extra sentences in file2's context not in file1's).
        added_context = find_added_sentences(sample_orig.get("context", ""), sample_adv.get("context", ""))

        output_sample = {
            "id": id_,
            "title": sample_orig.get("title", ""),
            "context": sample_orig.get("context", ""),
            "added_context": added_context,
            "question": sample_orig.get("question", ""),
            "original_answer": pred_orig,             # Predicted answer from file1.
            "adversarial_answer": pred_adv,             # Predicted answer from file2.
            "correct_answer": sample_orig.get("answers", {})  # The 'answers' field (a list/dict) from file1.
        }
        differing_examples.append(output_sample)

    if not differing_examples:
        print("No examples found where the predicted answers differ.")
        return

    # If -n is specified and is less than the total, randomly sample from the differing examples.
    if args.n is not None and args.n < len(differing_examples):
        random.seed(args.seed)
        differing_examples = random.sample(differing_examples, args.n)

    # Output each differing example in a human-readable multi-line format.
    for sample in differing_examples:
        print(f"id: {sample['id']}")
        print(f"title: {sample['title']}")
        print(f"context: {sample['context']}")
        print("added_context:")
        for sent in sample['added_context']:
            print(f"  - {sent}")
        print(f"question: {sample['question']}")
        print(f"original_answer (predicted from file1): {sample['original_answer']}")
        print(f"adversarial_answer (predicted from file2): {sample['adversarial_answer']}")
        print(f"correct_answer (from 'answers' field in file1): {sample['correct_answer']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
