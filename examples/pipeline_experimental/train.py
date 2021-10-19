import hetu as ht

import os
import sys
import time
import argparse
import numpy as np

from device_list import my_device, add_cpu_ctx
from resnet import resnet, get_resnet_partition
from utils import get_partition, get_tensorboard_writer, get_lr_scheduler

def validate(executor, val_batch_num):
    res = []
    for i in range(val_batch_num):
        res.append(executor.run("validate", convert_to_numpy_ret_vals=True))
    if res[0]:
        loss_value = []
        accuracy = []
        for iter_result in res:
            loss_value.append(iter_result[0][0])
            correct_prediction = np.equal(np.argmax(iter_result[1], 1), np.argmax(iter_result[2], 1)).mean()
            accuracy.append(correct_prediction)
        return np.mean(loss_value), np.mean(accuracy)
    return None, None

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--pipeline', type=str, default="pipedream")
    parser.add_argument('--preduce', action='store_true')
    parser.add_argument('--name', type=str, default="")
    args = parser.parse_args()

    assert args.pipeline in ["pipedream", "hetpipe"]
    assert args.dataset in ["cifar100", "imagenet"]
    assert args.model in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    ht.worker_init()
    np.random.seed(0)

    num_layers = 18
    model_partition = get_resnet_partition(num_layers)
    num_data_parallel = len(my_device[0])
    if args.pipeline == "hetpipe":
        assert not args.preduce
        my_device = add_cpu_ctx(my_device)
    device_list = get_partition(my_device, model_partition)
    args.learning_rate /= num_data_parallel

    if args.dataset == "cifar100":
        num_train_image, num_eval_image = 50000, 10000
    elif args.dataset == "imagenet":
        num_train_image, num_eval_image = 1281167, 50000
    train_batch_num = num_train_image // (args.batch_size * num_data_parallel)
    val_batch_num = num_eval_image // (args.batch_size * num_data_parallel)

    loss, y, y_ = resnet(args.dataset, args.batch_size, num_layers, device_list)

    if args.dataset == "cifar100":
        lr_scheduler = get_lr_scheduler(args.learning_rate, 0.2, 60 * train_batch_num, train_batch_num)
    elif args.dataset == "imagenet":
        lr_scheduler = get_lr_scheduler(args.learning_rate, 0.1, 30 * train_batch_num, train_batch_num)

    opt = ht.optim.SGDOptimizer(learning_rate=lr_scheduler, l2reg=args.weight_decay)
    with ht.context(device_list[-1]):
        train_op = opt.minimize(loss)
        executor = ht.Executor({"train" : [loss, y, y_, train_op], "validate" : [loss, y, y_]},
            seed=0, pipeline=args.pipeline, use_preduce=args.preduce)

    if executor.config.pipeline_dp_rank == 0:
        writer = get_tensorboard_writer(args.name)
    else:
        writer = None

    n_iter = 0

    for iteration in range(train_batch_num * args.epochs // args.log_every):
        start = time.time()
        res = executor.run("train", batch_num = args.log_every)
        if res[0]:
            time_used = time.time() - start
            loss_value = []
            accuracy = []
            for i, iter_result in enumerate(res):
                loss_value.append(iter_result[0][0])
                correct_prediction = np.equal(np.argmax(iter_result[1], 1), np.argmax(iter_result[2], 1)).mean()
                accuracy.append(correct_prediction)
                if writer:
                    writer.add_scalar('Train/loss', iter_result[0][0], n_iter + i)
                    writer.add_scalar('Train/acc', correct_prediction, n_iter + i)

            print(iteration, "TRAIN avg loss {:.4f}, acc {:.4f}, lr {:.4f}, time {:.4f}".format(
                    np.mean(loss_value), np.mean(accuracy), opt.learning_rate, time_used))

        n_iter += args.log_every

        val_loss, val_acc = validate(executor, val_batch_num)
        if val_loss:
            print(iteration, "EVAL avg loss {:.4f}, acc {:.4f}".format(val_loss, val_acc))
            if writer:
                writer.add_scalar('Validation/loss', val_loss, iteration)
                writer.add_scalar('Validation/acc', val_acc, iteration)
    ht.worker_finish()
