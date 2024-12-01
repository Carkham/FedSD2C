nc=10 # number of clients
dda=0.1 # dirichlet dataset alpha
md=ResNet18 # Conv5 or ResNet18
cuda=0 # cuda id
cmr=train_results/Baseline_Ensemble_dir${dda}_nc${nc}_ENSEMBLE_${md}_Imagenette_s42 # re-use the same client model root

python fedsd2c_main.py \
    -c configs/imagenette/fedsd2c.yaml \
    -dda $dda \
    -md $md \
    -is 42 \
    -sn FedSD2C_dir${dda}_nc${nc} \
    -g $cuda \
    -cmr $cmr \
    -nc $nc \
    -cis coreset+dist_syn \
    --fedsd2c_ipc 50 \
    --fedsd2c_inputs_init vae+fourier
