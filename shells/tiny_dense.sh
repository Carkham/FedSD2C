nc=10 # number of clients
dda=0.1 # dirichlet dataset alpha
md=Conv4 # Conv4 or ResNet18
cuda=0 # cuda id
cmr=train_results/Baseline_Ensemble_dir${dda}_nc${nc}_ENSEMBLE_${md}_TINYIMAGENET_s42  # re-use the same client model root

python oneshot_main.py \
    -c configs/tinyimagenet/dense.yaml \
    -dda $dda \
    -md $md \
    -is 42 \
    -sn Baseline_DENSE_dir${dda}_nc${nc} \
    -g $cuda \
    -nc $nc \
    -cmr $cmr
