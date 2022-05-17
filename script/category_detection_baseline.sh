cd ../
nohup python -u run.py \
    --exp 2 \
    --model 'fasterrcnn_resnet50' \
    --epoch 10 \
    --optim 'SGD' \
    --schedule 'Milestone' \
    --milestones 50 100 \
    --lr 0.001 \
    --gamma 0.1 \
    --train_batch 4 \
    --test_batch 4 \
    --gpu 1 \
    --memo '' \
    > results/nohup/category_detection_baseline.out &