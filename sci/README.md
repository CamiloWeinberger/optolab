# Spectral demosaicking

# Modes to train

python sci/pytorch/train.py --model Lightweight


python train.py --model light --model_type lowerloop --load_best_model 0
python train.py --model light --model_type lower --load_best_model 0


python train.py --model light --model_type lower --load_best_model 0 --sliding_window_approach 4 --use_gpus 1,2,3,4,5,6,7



python train.py --model eff3d --model_type s --load_best_model 0




python test.py --use_gpus 1,2,3,4,5,6,7 --save_results 0
