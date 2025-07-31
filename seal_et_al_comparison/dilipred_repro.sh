#!/bin/bash

OUT_FILE="dilipred_testing_predictions.csv"

echo "smiles,is_toxic" > $OUT_FILE
tail -n +2 "testing.csv" | while IFS=',' read -r SMILES _; do
    dilipred --smiles "$SMILES" --out temp.csv
    PRED=$(awk -F',' '$1 == "DILI" {print $4}' temp.csv)
    echo "$SMILES,$PRED" >> $OUT_FILE
done
rm temp.csv
