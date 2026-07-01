# Train - answer inference
CUDA_VISIBLE_DEVICES=0 python main.py --model ./models/unifiedqa-t5-base --user_msg answer --img_type detr --bs 4 --eval_bs 16 --eval_acc 10 --output_len 64 --final_eval --prompt_format QCMG-A --epoch 50 --vot_num 3 --alpha 0.5 --eval_le ./results/code6_train/rationale/predictions_ans_eval.json --test_le ./results/code6_train/rationale/predictions_ans_test.json --output_dir ./results/code6_train

# EVAL - answer inference
CUDA_VISIBLE_DEVICES=0 python main.py --model ./models/unifiedqa-t5-base --user_msg answer --img_type detr --bs 4 --eval_bs 16 --eval_acc 10 --output_len 64 --final_eval --prompt_format QCMG-A --eval_le ./results/code6_train/rationale/predictions_ans_eval.json --test_le ./results/code6_train/rationale/predictions_ans_test.json --evaluate_dir ./results/code6_train/answer
