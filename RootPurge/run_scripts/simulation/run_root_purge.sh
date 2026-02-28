export PYTHONPATH="RootPurge:$PYTHONPATH"
export PYTHONPATH="RootPurge/genTS:$PYTHONPATH"
export PYTHONPATH="RootPurge/genTS/external/tslib:$PYTHONPATH"



lr=1e-3
dropout_rate=0.0
n_train_step=20000
lookback_win=720
future_win=720
model_name=SpecLinear
indv=False


for seed in 2021 2022 2023 2024 2025; do
for t_end in 100 200 400 800 1600; do
for reg_lambda in 0.5 1 2 4; do
    echo "Running with prediction length $future_win"

    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path /dummy \
        --data_path dummy.csv \
        --model_id synthetic \
        --model $model_name \
        --data synthetic \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 64 \
        --d_model 64 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate \
        --lradj type4 \
        --data_ending $t_end \
        --data_noise 0.5 

done
done
done

for seed in 2021 2022 2023 2024 2025; do
for sig in 0 0.25 0.5 0.75 1; do
for reg_lambda in 0.5 1 2 4; do
    echo "Running with prediction length $future_win"

    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path /dummy \
        --data_path dummy.csv \
        --model_id synthetic \
        --model $model_name \
        --data synthetic \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 64 \
        --d_model 64 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate \
        --lradj type4 \
        --data_ending 400 \
        --data_noise $sig 

done
done
done


