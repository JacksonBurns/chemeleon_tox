OUT_FILE=dilipred_predictions.csv

echo "smiles,pred" > $OUT_FILE
tail -n +2 "recent_dili.csv" | while IFS=',' read -r _ _ SMILES; do
    dilipred --smiles "$SMILES" --out temp.csv
    PRED=$(awk -F',' '$1 == "DILI" {print $4}' temp.csv)
    echo "$SMILES,$PRED" >> $OUT_FILE
done
rm temp.csv
