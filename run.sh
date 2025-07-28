chemprop train \
    --data-path dilirank.csv \
    --output-dir chemprop_dilirank \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --target-columns dilirank \
    --task-type multiclass \
    --multiclass-num-classes 3 \
    --loss-function ce \
    --split-type random \
    --split-sizes 0.70 0.10 0.20 \
    --data-seed 42

python report.py --predictions-file chemprop_dilirank/model_0/test_predictions.csv --true-labels-file dilirank.csv > chemprop_dilirank/results.txt

chemprop train \
    --from-foundation CheMeleon \
    --data-path dilirank.csv \
    --output-dir chemeleon_dilirank \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --target-columns dilirank \
    --task-type multiclass \
    --multiclass-num-classes 3 \
    --loss-function ce \
    --split-type random \
    --split-sizes 0.70 0.10 0.20 \
    --data-seed 42

python report.py --predictions-file chemeleon_dilirank/model_0/test_predictions.csv --true-labels-file dilirank.csv > chemeleon_dilirank/results.txt

chemprop train \
    --data-path mitotox.csv \
    --output-dir mitotox_pretrain \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --from-foundation CheMeleon \
    --task-type classification \
    --class-balance \
    --split-type random \
    --split-sizes 0.90 0.09 0.01 \
    --data-seed 42

chemprop train \
    --data-path dilirank.csv \
    --output-dir mitotox_pretrain_dilirank_finetune \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --from-foundation mitotox_pretrain/model_0/best.pt \
    --task-type multiclass \
    --multiclass-num-classes 3 \
    --split-type random \
    --split-sizes 0.70 0.10 0.20 \
    --data-seed 42

python report.py --predictions-file mitotox_pretrain_dilirank_finetune/model_0/test_predictions.csv --true-labels-file dilirank.csv > mitotox_pretrain_dilirank_finetune/results.txt

chemprop train \
    --data-path mitotox.csv \
    --output-dir mitotox_pretrain_chemprop \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --task-type classification \
    --class-balance \
    --split-type random \
    --split-sizes 0.90 0.09 0.01 \
    --data-seed 42

chemprop train \
    --data-path dilirank.csv \
    --output-dir mitotox_pretrain_chemprop_dilirank_finetune \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --from-foundation mitotox_pretrain_chemprop/model_0/best.pt \
    --task-type multiclass \
    --multiclass-num-classes 3 \
    --split-type random \
    --split-sizes 0.70 0.10 0.20 \
    --data-seed 42

python report.py --predictions-file mitotox_pretrain_chemprop_dilirank_finetune/model_0/test_predictions.csv --true-labels-file dilirank.csv
