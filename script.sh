for seed in 1
do
    for modeltype in mlp mlpgpt
    do
        for testds in viecap sat
        do
            for trainds in viecap sat both
            do
                echo "Running seed $seed, trainds $trainds, testds $testds, modeltype $modeltype"
            done
        done
    done
done
