export PYTHONPATH="RootPurge:$PYTHONPATH"
export PYTHONPATH="RootPurge/genTS:$PYTHONPATH"
export PYTHONPATH="RootPurge/genTS/external/tslib:$PYTHONPATH"


lr=5e-4
dropout_rate=0.0
n_train_step=5000
lookback_win=720
model_name=SpecLinear
indv=False


for seed in 2021 2022 2023 2024 2025; do
for reg_lambda in 0.0 0.125 0.25 0.5; do
for future_win in 720 336 192 96; do
    echo "Running with prediction length $future_win"


    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path datasets/time_series/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 64 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate

    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path datasets/time_series/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 64 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate


    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path datasets/time_series/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 64 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate


    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path datasets/time_series/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 64 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate



    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path datasets/time_series/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 64 \
        --d_model 64 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate


    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path datasets/time_series/traffic/ \
        --data_path traffic.csv \
        --model_id traffic \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 64 \
        --d_model 64 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate


    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path datasets/time_series/weather/ \
        --data_path weather.csv \
        --model_id weather \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 64 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef $reg_lambda \
        --seed $seed \
        --individual True \
        --dropout $dropout_rate



    python -u linear_model/run_forecasting.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path datasets/time_series/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id exchange_rate \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 64 \
        --d_model 64 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --learning_rate 5e-3 \
        --train_steps 2500 \
        --regu_coef $reg_lambda \
        --seed $seed \
        --dropout $dropout_rate

done
done
done