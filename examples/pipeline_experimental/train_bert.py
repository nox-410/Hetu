import hetu as ht

import os
import sys
import time
import argparse
import numpy as np

from device_list import my_device, add_cpu_ctx
from models.hetu_bert import BertForPreTraining
from models.bert_config import BertConfig
from utils import get_partition, get_tensorboard_writer, get_lr_scheduler
from reduce_result import reduce_result

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
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--pipeline', type=str, default="pipedream")
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--preduce', action='store_true')
    parser.add_argument('--name', type=str, default="")
    args = parser.parse_args()

    assert args.pipeline in ["pipedream", "hetpipe", "dp"]
    if args.pipeline == "dp":
        args.pipeline = None # use common data parallel
    ht.worker_init()
    np.random.seed(0)
    model_partition = [2, 4, 4, 2]
    num_data_parallel = len(my_device[0])
    if args.pipeline == "hetpipe":
        assert not args.preduce
        my_device = add_cpu_ctx(my_device)
    # Note : when we use model average, we don't need to modify lr
    # In allreduce case, we use reduce mean, otherwise we have to modify weight decay param
    device_list = get_partition(my_device, model_partition)
    config = BertConfig(vocab_size=30522,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    max_position_embeddings=512,
                    # attention_probs_dropout_prob=0.0,
                    # hidden_dropout_prob=0.0,
                    batch_size=args.batch_size)

    with ht.context(ht.ndarray.gpu(1)):
        model = BertForPreTraining(config=config)
        _,_,masked_lm_loss_mean, next_sentence_loss_mean, loss = model()

    opt = ht.optim.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = 0.01)
    with ht.context(ht.ndarray.gpu(1)):
        train_op = opt.minimize(loss)
        executor = ht.Executor({"train" : [loss, masked_lm_loss_mean, next_sentence_loss_mean, train_op]},
            seed=0, pipeline=args.pipeline, use_preduce=args.preduce, dynamic_memory=True)

    if executor.config.pipeline_dp_rank == 0:
        writer = get_tensorboard_writer(args.name)
    else:
        writer = None

    if executor.rank == 0:
        print("Training {} epoch, each epoch runs {} iteration".format(args.epochs, 100))

    for iteration in range(args.epochs):
        start = time.time()
        if args.pipeline:
            res = executor.run("train", batch_num = 100)
        elif args.preduce:
            res = []
            while not executor.subexecutor["train"].preduce_stop_flag:
                res.append(executor.run("train", convert_to_numpy_ret_vals=True))
            executor.subexecutor["train"].preduce_stop_flag = False
        else:
            res = []
            for i in range(100):
                res.append(executor.run("train", convert_to_numpy_ret_vals=True))
        if res[0]:
            time_used = time.time() - start
            loss_value = []
            accuracy = []
            for i, iter_result in enumerate(res):
                loss_value.append(iter_result[0][0])
                print(iter_result[0][0])
                # correct_prediction = np.equal(np.argmax(iter_result[1], 1), np.argmax(iter_result[2], 1)).mean()
                # accuracy.append(correct_prediction)
            # loss_value = reduce_result([np.mean(loss_value)])
            # if writer:
            #     writer.add_scalar('Train/loss', loss_value, iteration)
            #     writer.add_scalar('Train/acc', accuracy, iteration)
            # if args.preduce:
            #     preduce_mean = executor.subexecutor["train"].preduce.mean
            #     print(preduce_mean)
            #     executor.subexecutor["train"].preduce.reset_mean()
            # if executor.config.pipeline_dp_rank == 0:
            #     print(iteration, "TRAIN loss {:.4f}  lr {:.2e}, time {:.4f}".format(
            #         loss_value, opt.learning_rate, time_used))

    ht.worker_finish()
