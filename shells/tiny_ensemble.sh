nc=10 # number of clients
dda=0.1 # dirichlet dataset alpha
md=Conv4 # Conv4 or ResNet18
cuda=0 # cuda id

python oneshot_main.py \
    -c configs/imagenette/ensemble.yaml \
    -dda $dda \
    -md $md \
    -is 42 \
    -sn Baseline_Ensemble_dir${dda}_nc${nc} \
    -g $cuda \
    -nc $nc \
    --save_client_model
