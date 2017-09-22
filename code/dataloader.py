import sys
import threading
import queue
import random

import torch
import torch.multiprocessing as multiprocessing

from torch.utils.data.dataloader import _use_shared_memory
from torch.utils.data.dataloader import ExceptionWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import _pin_memory_loop

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale):
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            scaleIdx = 0
            if len(scale) > 1 and dataset.train:
                scaleIdx = random.randrange(0, len(scale))
                dataset.setScale(scaleIdx)
            
            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(scaleIdx)
            
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))

class MSDataLoaderIter(DataLoaderIter):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_ms_loop,
                    args=(
                        self.dataset,
                        self.index_queue, self.data_queue,
                        self.collate_fn, self.scale))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

class MSDataLoader(DataLoader):
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        pin_memory=False, drop_last=False):

        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.nThreads, collate_fn=default_collate,
            pin_memory=pin_memory, drop_last=drop_last)

        self.scale = args.scale

    def __iter__(self):
        return MSDataLoaderIter(self)
