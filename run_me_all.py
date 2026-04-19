from kernel import main
from train_args import args_list

target_rate = 3



def func(args, path_num, speed):
    return main(
        args=args,
        seed=1052026,
        speed=speed,
        target_rate=target_rate,
        ROOT_PATH=f'./output/speed_{speed}/{path_num}'
    )

if __name__ == "__main__":
    import os, shutil



    if os.path.exists('./output'):
        shutil.rmtree('./output')

    speed = 10

    for i, args in enumerate(args_list):
        func(args, i, speed)
