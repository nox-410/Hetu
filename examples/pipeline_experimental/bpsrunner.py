import argparse
import yaml
import os
import signal
import multiprocessing
import subprocess
import paramiko
import socket
import psutil
import contextlib
import hetu as ht
from itertools import zip_longest

_procs = []


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in _procs:
        proc.kill()
    exit(0)

@ contextlib.contextmanager
def ssh_connect(host, identify_file):
    try:
        ssh_directory = os.path.expanduser('~/.ssh') if identify_file == '' else os.path.dirname(
            os.path.abspath(os.path.expanduser(identify_file)))
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        private = paramiko.RSAKey.from_private_key_file(
            os.path.join(ssh_directory, 'id_rsa'))
        config = paramiko.config.SSHConfig.from_path(
            os.path.join(ssh_directory, 'config'))
        conf = config.lookup(host)
        ssh.connect(hostname=conf['hostname'], port=conf['port'],
                    username=conf['user'], pkey=private)
        yield ssh
    finally:
        ssh.close()


def start_server(host, identify_file):
    with ssh_connect(host, identify_file) as ssh:
        stdin, stdout, stderr = ssh.exec_command("DMLC_ROLE=server DMLC_PS_ROOT_URI={} DMLC_PS_ROOT_PORT={} \
DMLC_NUM_WORKER={} DMLC_NUM_SERVER={} bpslaunch".format(os.environ["DMLC_PS_ROOT_URI"], os.environ["DMLC_PS_ROOT_PORT"],
    os.environ["DMLC_NUM_WORKER"], os.environ["DMLC_NUM_SERVER"]))
        stdout_iter = iter(stdout.readline, '')
        stderr_iter = iter(stderr.readline, '')

        for out in stdout_iter:
            print("STDOUT scheduler", out.strip())
        for err in stderr_iter:
            print("STDERR scheduler", err.strip())

def start_scheduler(host, identify_file):
    with ssh_connect(host, identify_file) as ssh:
        stdin, stdout, stderr = ssh.exec_command("DMLC_ROLE=scheduler DMLC_PS_ROOT_URI={} DMLC_PS_ROOT_PORT={} \
DMLC_NUM_WORKER={} DMLC_NUM_SERVER={} bpslaunch".format(os.environ["DMLC_PS_ROOT_URI"], os.environ["DMLC_PS_ROOT_PORT"],
    os.environ["DMLC_NUM_WORKER"], os.environ["DMLC_NUM_SERVER"]))
        stdout_iter = iter(stdout.readline, '')
        stderr_iter = iter(stderr.readline, '')

        for out in stdout_iter:
            print("STDOUT server", out.strip())
        for err in stderr_iter:
            print("STDERR server", err.strip())

def start_worker(host, identify_file, worker_id, command):
    with ssh_connect(host, identify_file) as ssh:
        stdin, stdout, stderr = ssh.exec_command("cd {} && DMLC_ROLE=worker DMLC_PS_ROOT_URI={} DMLC_PS_ROOT_PORT={} \
DMLC_NUM_WORKER={} DMLC_NUM_SERVER={} DMLC_WORKER_ID={} bpslaunch {}".format(os.environ["PWD"], os.environ["DMLC_PS_ROOT_URI"], os.environ["DMLC_PS_ROOT_PORT"],
    os.environ["DMLC_NUM_WORKER"], os.environ["DMLC_NUM_SERVER"], worker_id, command))
        stdout_iter = iter(stdout.readline, '')
        stderr_iter = iter(stderr.readline, '')

        for out in stdout_iter:
            print("STDOUT worker", worker_id, out.strip())
        for err in stderr_iter:
            print("STDERR worker", worker_id, err.strip())

def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None,
                        help='Configuration file.')
    parser.add_argument('-w', '--workers', type=int, default=0,
                        help='Shorthand for the number of local worker.')
    parser.add_argument('-s', '--servers', type=int, default=0,
                        help='Shorthand for the number of local server.')
    parser.add_argument('-i', '--identify', default='',
                        help='SSH identify file.')
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Command to be executed.')
    args = parser.parse_args()
    settings = ht.dist.DistConfig(args.config, args.servers, args.workers)
    print(settings)
    ps_config = settings.make_ps_config()
    for k, v in ps_config.items():
        os.environ[k] = str(v)

    proc = multiprocessing.Process(target=start_scheduler, args=[settings.chief, args.identify])
    _procs.append(proc)

    for node in settings.hosts:
        if settings.servers.get(node, 0):
            for _ in range(settings.servers[node]):
                proc = multiprocessing.Process(target=start_server, args=[
                                            node, args.identify])
                _procs.append(proc)
        if settings.workers.get(node, 0):
            for i in range(settings.workers[node]):
                proc = multiprocessing.Process(target=start_worker, args=[
                                            node, args.identify, i, ' '.join(args.command)])
                _procs.append(proc)
    for proc in _procs:
        proc.start()
    for proc in _procs:
        proc.join()

if __name__ == '__main__':
    main()
