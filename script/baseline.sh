cd ../
nohup python -u run.py \
    --exp 1 \
    --model 'resnet34' \
    --epoch 4 \
    --optim 'SGD' \
    --lr 0.001 \
    --train_batch 100 \
    --test_batch 100 \
    --gpu 0 \
    --memo '' \
    > results/nohup/baseline.out &

