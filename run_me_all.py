from kernel import main
from train_args import args_list
import matplotlib.pyplot as plt

target_rate = 1200  # Mbps

def func(args, path_num, speed):
    result = main(args=args, seed=20205598, speed=speed, target_rate=target_rate, 
                  ROOT_PATH='./output/' + 'speed_' + str(speed) + '/' + str(path_num))
    return result

if __name__ == "__main__":
    import os
    import shutil

    if os.path.exists('./output'):
        shutil.rmtree('./output')

    times = len(args_list)
    speed = 10

    all_results = []
    for i in range(times):
        result = func(args_list[i], i, speed)
        all_results.append((i, speed, result))
        # print(f"Result for path {i}, speed {sped}: {result}")
