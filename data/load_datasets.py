"""Load and inspect the CNN/DailyMail and XSum datasets."""
from datasets import load_dataset, get_dataset_split_names


def load_cnn_dailymail(split="train", num_samples=None):
    """CNN/DailyMail v3.0.0. Columns: article, highlights, id."""
    ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
    if num_samples:
        ds = ds.select(range(num_samples))
    return ds


def load_xsum(split="train", num_samples=None):
    """XSum. Columns: document, summary, id."""
    ds = load_dataset("xsum", split=split)
    if num_samples:
        ds = ds.select(range(num_samples))
    return ds


def get_columns(dataset_name):
    """Map dataset name -> (input_col, target_col)."""
    if dataset_name == "cnn_dailymail":
        return "article", "highlights"
    return "document", "summary"


if __name__ == "__main__":
    # Quick sanity checks
    for name in ["cnn_dailymail", "xsum"]:
        ds = load_cnn_dailymail() if name == "cnn_dailymail" else load_xsum()
        print(f"{name}: {len(ds)} train examples")
        print("  splits:", get_dataset_split_names(name))
        print("  columns:", ds.column_names)
        print("  sample input:", ds[0][get_columns(name)[0]][:120], "...")
        print("  sample target:", ds[0][get_columns(name)[1]][:120], "...")
