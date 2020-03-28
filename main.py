import os
import time

from algorithm.BSP import BSP
from ps.DatasetFactory import DatasetFactory

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    num_workers = 3
    sleep_gap = 0.1
    iterations = 200
    min_lock = 5
    dataset_factory = DatasetFactory(
        data_root='/Users/dmrfcoder/Documents/毕业设计/code/nuaa_distributed_machine_learning/mnist/data')

    data_loaders = dataset_factory.build_dataset(num_workers,shuffle=False)
    bsp = BSP(sleep_gap=sleep_gap, data_loaders=data_loaders, num_workers=num_workers, dataset_factory=dataset_factory,
              iterations=iterations)
    # asp = ASP(sleep_gap=sleep_gap, data_loaders=data_loaders, num_workers=num_workers, dataset_factory=dataset_factory,
    #           iterations=iterations)
    # ssp = SSP(sleep_gap=sleep_gap, data_loaders=data_loaders, num_workers=num_workers, dataset_factory=dataset_factory,
    #           iterations=iterations, min_lock=min_lock)

    starttime = time.time()
    bsp.execute(show_plt=True)
    endtime = time.time()
    print('任务总耗时：' + str(endtime - starttime) + ' s')
