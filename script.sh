for seed in 1
do
    for modeltype in mlp mlpgpt
    do
        for valds in viecap sat
        do
            for trainds in viecap sat both
            do
                echo "Running seed $seed, trainds $trainds, valds $valds, modeltype $modeltype"

                XRT_TPU_CONFIG="localservice;0;localhost:51011" accelerate launch \
                    --config_file ./config_tpu.yml \
                    train.py \
                    --epochs 30 \
                    --save-every 1 \
                    --val-every 1 \
                    --lr "1e-3" \
                    --batch-size 32 \
                    --train-dataset "${trainds}" \
                    --val-dataset "${valds}" \
                    --model-type "${modeltype}" \
                    --seed "${seed}"
            done
        done
    done
done
