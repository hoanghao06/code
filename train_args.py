import argparse

def my_args(params):
    parser = argparse.ArgumentParser("PPO")

    parser.add_argument("--max_train_episodes", type=int, default=params["max_train_episodes"])
    parser.add_argument("--evaluate_episode_freq", type=int, default=params["evaluate_episode_freq"])
    parser.add_argument("--policy_dist", type=str, default="Beta")

    parser.add_argument("--batch_size", type=int, default=params['batch_size'])
    parser.add_argument("--mini_batch_size", type=int, default=params['mini_batch_size'])

    parser.add_argument("--hidden_width", type=int, default=params["hidden_width"])
    parser.add_argument("--lr_a", type=float, default=params['lr_a'])
    parser.add_argument("--lr_c", type=float, default=params['lr_c'])
    parser.add_argument("--car_num", type=int, default=3)

    parser.add_argument("--gamma", type=float, default=params['gamma'])
    parser.add_argument("--lamda", type=float, default=params['lamda'])
    parser.add_argument("--epsilon", type=float, default=params['epsilon'])
    parser.add_argument("--K_epochs", type=int, default=8)

    parser.add_argument("--use_adv_norm", type=bool, default=params['use_adv_norm'])
    parser.add_argument("--use_state_norm", type=bool, default=params['use_state_norm'])

    parser.add_argument("--entropy_coef", type=float, default=params['entropy_coef'])
    parser.add_argument("--use_lr_decay", type=bool, default=params['use_lr_decay'])
    parser.add_argument("--use_grad_clip", type=bool, default=params['use_grad_clip'])

    parser.add_argument("--set_adam_eps", type=bool, default=params['set_adam_eps'])
    parser.add_argument("--use_tanh", type=bool, default=params['use_tanh'])

    return parser.parse_args([])  


env_steps = 300

arg_dict_0 = {
    "max_train_episodes": 3000,
    "evaluate_episode_freq": 15,
    "batch_size": env_steps * 3,
    "mini_batch_size": env_steps // 3,
    "hidden_width": 256,
    "lr_a": 2e-4,
    "lr_c": 4e-4,
    "gamma": 0.99,
    "lamda": 0.95,
    "epsilon": 0.15,
    "use_adv_norm": True,
    "use_state_norm": True,
    "entropy_coef": 0.005,
    "use_lr_decay": True,
    "use_grad_clip": True,
    "set_adam_eps": True,
    "use_tanh": True
}

arg_dict_1 = {
    "max_train_episodes": 3000,
    "evaluate_episode_freq": 15,
    "batch_size": env_steps * 3,
    "mini_batch_size": env_steps // 3,
    "hidden_width": 128,
    "lr_a": 2e-4,
    "lr_c": 4e-4,
    "gamma": 0.99,
    "lamda": 0.95,
    "epsilon": 0.2,
    "use_adv_norm": True,
    "use_state_norm": True,
    "entropy_coef": 0.005,
    "use_lr_decay": True,
    "use_grad_clip": True,
    "set_adam_eps": True,
    "use_tanh": True
}

arg_dict_2 = {
    "max_train_episodes": 3000,
    "evaluate_episode_freq": 10,
    "batch_size": env_steps * 3,
    "mini_batch_size": env_steps // 3,
    "hidden_width": 128,
    "lr_a": 2e-4,
    "lr_c": 4e-4,
    "gamma": 0.98,
    "lamda": 0.98,
    "epsilon": 0.2,
    "use_adv_norm": True,
    "use_state_norm": True,
    "entropy_coef": 0.005,
    "use_lr_decay": True,
    "use_grad_clip": True,
    "set_adam_eps": True,
    "use_tanh": True
}

arg_dict_3 = {
    "max_train_episodes": 3000,
    "evaluate_episode_freq": 10,
    "batch_size": env_steps * 3,
    "mini_batch_size": env_steps // 3,
    "hidden_width": 128,
    "lr_a": 2e-4,
    "lr_c": 4e-4,
    "gamma": 0.98,
    "lamda": 0.98,
    "epsilon": 0.15,
    "use_adv_norm": True,
    "use_state_norm": True,
    "entropy_coef": 0.005,
    "use_lr_decay": True,
    "use_grad_clip": True,
    "set_adam_eps": True,
    "use_tanh": True
}


# args_list = [my_args(arg_dict_2)]

temp_args_list = [ arg_dict_3]

args_list = []

for item in temp_args_list:
    args_list.append(my_args(item))
