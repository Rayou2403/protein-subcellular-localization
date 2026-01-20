"""
Prepare metadata CSV from DeepLocPro FASTA files.

The FASTA headers have format: >PROTEIN_ID|LOCATION|GRAM_TYPE|PARTITION
Example: >A0A0C5CJR8|Extracellular|negative|1

This script extracts this information and creates a CSV file with columns:
- protein_id
- sequence
- label (subcellular location)
- gram_type (positive/negative)
- partition (GraphPart cluster ID)
"""

import argparse
import csv
from Bio import SeqIO
from pathlib import Path


def parse_fasta_to_metadata(fasta_file: str, output_csv: str):
    """
    Parse FASTA file and create metadata CSV.

    Args:
        fasta_file: Input FASTA file path
        output_csv: Output CSV file path
    """
    records = []

    with open(fasta_file, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            header_parts = rec.id.split("|")

            if len(header_parts) < 3:
                print(f"WARNING: Skipping malformed header: {rec.id}")
                continue

            protein_id = header_parts[0]
            location = header_parts[1]
            gram_type = header_parts[2]
            partition = header_parts[3] if len(header_parts) > 3 else "0"
            sequence = str(rec.seq)

            records.append({
                "protein_id": protein_id,
                "sequence": sequence,
                "label": location,
                "gram_type": gram_type,
                "partition": partition,
            })

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["protein_id", "sequence", "label", "gram_type", "partition"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(f"Saved {len(records)} protein records to {output_csv}")

    locations = {}
    for rec in records:
        loc = rec["label"]
        locations[loc] = locations.get(loc, 0) + 1

    print("\nClass distribution:")
    for loc, count in sorted(locations.items()):
        print(f"  {loc}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare metadata CSV from DeepLocPro FASTA file"
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file (e.g., data/graphpart_set.fasta)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file (e.g., data/metadata.csv)",
    )
    args = parser.parse_args()

    parse_fasta_to_metadata(args.fasta, args.output)


if __name__ == "__main__":
    main()
