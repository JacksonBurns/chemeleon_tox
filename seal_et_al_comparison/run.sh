COMMON_TRAIN_ARGS="
    --ffn-num-layers 2 \
    --ffn-hidden-dim 1800 \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --target-columns is_toxic \
    --task-type classification \
    --class-balance \
    --data-seed 42 \
    --patience 5 \
    --num-workers 4
"

COMMON_TEST_ARGS="
    --test-path testing.csv \
    --smiles-column smiles \
    --num-workers 4
"

# there are three reasonable ways to approach this modeling problem:
# 1) train on the highest quality, lowest quantity data
# 2) train on the lowest quality, highest quantity data
# 3) pretrain on the low quality data first, then finetune on the high quality
#
# we will try all three to compare their performance

# 1)
chemprop train \
    --from-foundation CheMeleon \
    --data-path finetuning.csv \
    --output-dir outputs/strategy_1 \
    --split-type random \
    --split-sizes 0.90 0.10 0.00 \
    $COMMON_TRAIN_ARGS

chemprop predict \
    --model-path outputs/strategy_1/model_0/best.pt \
    --preds-path outputs/strategy_1/test_predictions.csv \
    $COMMON_TEST_ARGS

# 2)
chemprop train \
    --from-foundation CheMeleon \
    --data-path pretraining.csv \
    --output-dir outputs/strategy_2 \
    --split-type random \
    --split-sizes 0.90 0.10 0.00 \
    $COMMON_TRAIN_ARGS

chemprop predict \
    --model-path outputs/strategy_2/model_0/best.pt \
    --preds-path outputs/strategy_2/test_predictions.csv \
    $COMMON_TEST_ARGS

# 3)
chemprop train \
    --data-path finetuning.csv \
    --output-dir outputs/strategy_3 \
    --checkpoint outputs/strategy_2/model_0/best.pt \
    --split-type random \
    --split-sizes 0.90 0.10 0.00 \
    $COMMON_TRAIN_ARGS

chemprop predict \
    --model-path outputs/strategy_3/model_0/best.pt \
    --preds-path outputs/strategy_3/test_predictions.csv \
    $COMMON_TEST_ARGS
