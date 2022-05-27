cd ../
nohup python -u run.py \
    --exp 1 \
    --model 'wide_resnet50_2' \
    --use_val 0 \
    --eval_every 0 \
    --epoch 5 \
    --optim 'SGD' \
    --gamma 0.1 \
    --schedule Milestone \
    --milestones 2 4 \
    --lr_base 0.001 \
    --lr_cf 0.01 \
    --train_batch 64 \
    --test_batch 64 \
    --gpu 0 \
    --strategy 'Cumulative' \
    --use_cutmix 'True' \
    --memo '' \
    > results/nohup/wide_resnet50_2-cutmix.out &