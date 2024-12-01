import argparse


def fed_args():
    """
    Arguments for running federated learning baselines
    :return: Arguments for federated learning baselines
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default="configs/ensemble.yaml", help='path to config file;',)
    parser.add_argument('-sn', '--save_name', type=str, required=True, help='file name for log, statistic and checkpoint')
    parser.add_argument('-g', '--gpu_id', type=str, default="cuda:0", help="Gpu id")
    parser.add_argument('--using_wandb', action="store_true", default=False)
    parser.add_argument('--wandb_proj_name', type=str)

    parser.add_argument('-nc', '--sys-n_client', type=int, help='Number of the clients')
    parser.add_argument('-ds', '--sys-dataset', type=str,
                        help='Dataset name, one of the following four datasets: MNIST, CIFAR-10, FashionMnist, SVHN')
    parser.add_argument('-md', '--sys-model', type=str, help='Model name')
    parser.add_argument('-is', '--sys-i_seed', type=int, help='Seed used in experiments')
    parser.add_argument('-rr', '--sys-res_root', type=str, help='Root directory of the results')
    parser.add_argument('-dda', '--sys-dataset_dir_alpha', type=float, help='Alpha used for partitioning dataset with dirichlet')

    parser.add_argument('-sne', '--server-n_epoch', type=int,
                        help='Number of training epochs in the server')
    parser.add_argument('-sbs', '--server-bs', type=int, help='Batch size in the server')
    parser.add_argument('-slr', '--server-lr', type=float, help='Learning rate in the server')
    parser.add_argument('-smt', '--server-momentum', type=float, help='Momentum in the server')
    parser.add_argument('-snw', '--server-n_worker', type=int, help='Number of workers in the server')
    parser.add_argument('-so', '--server-optimizer', type=str, help='Optimizer for server model')
    parser.add_argument('-sls', '--server-lr_scheduler', type=str, default="cos", help='Lr scheduler in the server')

    parser.add_argument('-cis', '--client-instance', type=str,
                        help='Instance of federated learning algorithm used in clients')
    parser.add_argument('-cil', '--client-instance_lr', type=float, help='Learning rate in clients')
    parser.add_argument('-cib', '--client-instance_bs', type=int, help='Batch size in clients')
    parser.add_argument('-cie', '--client-instance_n_epoch', type=int,
                        help='Number of local training epochs in clients')

    # load trained client model
    parser.add_argument('-cmr', '--client_model_root', type=str, default=None, help='If not None load client model from the root')
    parser.add_argument('--client-instance_mixup_alpha', type=float, default=0.0, help='Mixup alpha for local training')
    parser.add_argument('--client-instance_cutmix_alpha', type=float, default=0.0, help='Cutmix alpha for local training')

    parser.add_argument('--save_client_model', action="store_true", default=False, help='Whether to save the trained client models to the sys-res_root')

    # FedSD2C
    parser.add_argument('-fm', '--fedsd2c_mipc', type=int, help='Number of pre-loaded images per class')
    parser.add_argument('-fi', '--fedsd2c_ipc', type=int, help='IPC for client dataset distillation')
    parser.add_argument('-fnc', '--fedsd2c_num_crop', type=int, help='Number of crop for CoreSet selection')

    # Loss
    parser.add_argument('-fii', '--fedsd2c_inputs_init', default="vae+fourier", type=str, help='Initialization method for the input images')
    parser.add_argument('-fj', '--fedsd2c_jitter', default=0, type=int, help='jitter')
    parser.add_argument('-fit', '--fedsd2c_iteration', default=100, type=int, help='Number of synthesis steps')
    parser.add_argument('-flr', '--fedsd2c_lr', default=0.1, type=float, help='lr for synthesis stage')
    parser.add_argument('-fls', '--fedsd2c_l2_scale', type=float, default=0, help='coefficient for synthesis l2 loss')
    parser.add_argument('-ftl', '--fedsd2c_tv_l2', type=float, default=0, help='coefficient for synthesis tv l2 loss')
    parser.add_argument('-frb', '--fedsd2c_r_bn', type=float, default=0, help='coefficient for synthesis bn loss')
    parser.add_argument('-frc', '--fedsd2c_r_c', type=float, default=0, help='coefficient for other client ce loss')
    parser.add_argument('-fra', '--fedsd2c_r_adv', type=float, default=0, help='coefficient for adversarial los')
    parser.add_argument('--fedsd2c_loss', type=str, default="mse", help='Loss function for distillation')
    parser.add_argument('--fedsd2c_iter_mode', type=str, default="label", choices=["label", "ipc", "score"], help='Iteration mode for distillation')

    # Adding noise on latents
    parser.add_argument('--fedsd2c_noise_type', type=str, default="None")
    parser.add_argument('--fedsd2c_noise_s', type=float, default=0., help="scale of the noise")
    parser.add_argument('--fedsd2c_noise_p', type=float, default=0., help="proportion of the noise")
    parser.add_argument('--fourier_lambda', type=float, default=0.8, help="lambda for fourier transform perturbation")

    # FedCVAE
    parser.add_argument('--cvae_z_dim', type=int)
    parser.add_argument('--cvae_beta', type=float)

    # DFKD
    parser.add_argument('--dfkd_z_dim', type=int)
    parser.add_argument('--dfkd_temp', type=float)
    parser.add_argument('--dfkd_r_bn', type=float)
    parser.add_argument('--dfkd_r_adv', type=float)
    parser.add_argument('--dfkd_r_bal', type=float)
    parser.add_argument('--dfkd_r_oh', type=float)
    parser.add_argument('--dfkd_giter', type=int)
    parser.add_argument('--dfkd_miter', type=int)
    parser.add_argument('--dfkd_eiter', type=int)
    parser.add_argument('--dfkd_img_size', type=int)
    parser.add_argument('--dfkd_batch_size', type=int)
    parser.add_argument('--dfkd_syn_data', type=bool, default=True)


    # CoBoost
    parser.add_argument('--cb_z_dim', type=int)
    parser.add_argument('--cb_temp', type=float)
    parser.add_argument('--cb_r_bn', type=float)
    parser.add_argument('--cb_div', type=float)
    parser.add_argument('--cb_giter', type=int)
    parser.add_argument('--cb_miter', type=int)
    parser.add_argument('--cb_witer', type=int)
    parser.add_argument('--cb_hs', type=float)
    parser.add_argument('--cb_oh', type=float)
    parser.add_argument('--cb_mu', type=float)
    parser.add_argument('--cb_wdc', type=float)
    parser.add_argument('--cb_weighted', type=bool, default=False)
    parser.add_argument('--cb_odseta', type=float)

    args = parser.parse_args()
    return args
