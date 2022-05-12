cd ../
nohup python -u run.py \
    --exp 1 \
    --model 'resnet34' \
    --use_val 1 \
    --eval_every 1 \
    --epoch 220 \
    --optim 'SGD' \
    --gamma 0.1 \
    --schedule Milestone \
    --milestones 80 160 \
    --lr_base 0.001 \
    --lr_cf 0.01 \
    --train_batch 256 \
    --test_batch 256 \
    --gpu 0 \
    --memo '' \
    > results/nohup/classification_baseline.out &

