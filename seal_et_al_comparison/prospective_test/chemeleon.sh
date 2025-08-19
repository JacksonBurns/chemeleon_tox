chemprop predict \
    --model-path ../outputs/strategy_3/model_0/best.pt \
    --preds-path chemeleon_predictions.csv \
    --test-path recent_dili.csv \
    --smiles-column smiles \
    --num-workers 4
