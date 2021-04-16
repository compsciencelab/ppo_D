Official implementation of the paper "Guided Exploration with Proximal PolicyOptimization using a Single Demonstration" https://arxiv.org/pdf/2007.03328.pdf

Test run for lunar lander

python train_ppo_bullet.py --seed 16 --device 'cuda:0' --use-gae --lr 2e-4 --clip-param 0.2 --value-loss-coef 0.3 --num-processes 12 --num-steps 2048 --num-mini-batch 32 --entropy-coef 0.02 --num-env-steps 60000000 --log-dir ../RUNS/exp_lunar_lander_1 --frame-stack 1  --cnn MLP  --gamma 0.99 --save-interval 50 --gae-lambda 0.95 --ppo-epoch 10 --state-stack 16 --rho 0.1 --phi 0.0  --size-buffer 50 --size-buffer-V 0 --demo-dir ../datasets/sparse_lunar_lander/recordings/ --threshold-reward 0.0 --task 'SparseLunarLander-v1'

