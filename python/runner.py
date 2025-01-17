import argparse
import yaml
import os
import signal
import multiprocessing
import subprocess
import paramiko
import socket
import psutil
import hetu as ht

_procs = []


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in _procs:
        proc.kill()
    global executor_shell
    executor_shell.kill()
    exit(0)


def start_sched():
    os.environ["DMLC_ROLE"] = "scheduler"
    ht.scheduler_init()
    ht.scheduler_finish()


def start_server():
    os.environ["DMLC_ROLE"] = "server"
    ht.server_init()
    ht.server_finish()


def start_remote_server(host, local_server_num, identify_file):
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
    sftp = ssh.open_sftp()
    sftp.put('/tmp/temp_hetu_config.yml',
             '/tmp/temp_hetu_config.yml', confirm=True)
    sftp.close()
    stdin, stdout, stderr = ssh.exec_command(
        'python -m hetu.launcher /tmp/temp_hetu_config.yml -n %d' % local_server_num)
    stdout = stdout.read().decode()
    stderr = stderr.read().decode()
    if stdout:
        print('From remote %s stdout:\n %s' % (host, stdout.strip()))
    if stderr:
        print('From remote %s stderr:\n %s' % (host, stderr.strip()))
    ssh.close()


def get_available_port(localhost):
    ports = set()
    for conn in psutil.net_connections():
        la = conn.laddr
        ra = conn.raddr
        if len(la) == 2 and la.ip in (localhost, '127.0.0.1'):
            ports.add(la.port)
        if len(ra) == 2 and ra.ip in (localhost, '127.0.0.1'):
            ports.add(ra.port)
    for p in range(13100, 13200):
        if p not in ports:
            return p


def get_nic_names(local_address, remote_hostnames, identify_file):
    # get local interface
    nics = dict()
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                nics[addr.address] = iface
    local_nic = nics[local_address]

    # get remote interfaces
    command_prefix = "\"from socket import AF_INET;\nfrom psutil import net_if_addrs;\n" +\
        "nics = dict();\nfor iface, addrs in net_if_addrs().items():\n    for addr in addrs:" +\
        "\n        if addr.family == AF_INET:\n            nics[addr.address] = iface;\n"
    ssh_directory = os.path.expanduser('~/.ssh') if identify_file == '' else os.path.dirname(
        os.path.abspath(os.path.expanduser(identify_file)))
    remote_nics = set()
    for hostname in remote_hostnames:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        private = paramiko.RSAKey.from_private_key_file(
            os.path.join(ssh_directory, 'id_rsa'))
        config = paramiko.config.SSHConfig.from_path(
            os.path.join(ssh_directory, 'config'))
        conf = config.lookup(hostname)
        command = command_prefix + "print(nics[\'%s\'])\"" % (conf['hostname'])
        ssh.connect(hostname=conf['hostname'], port=conf['port'],
                    username=conf['user'], pkey=private)
        stdin, stdout, stderr = ssh.exec_command('python -c %s' % command)
        stdout = stdout.read().decode()
        stderr = stderr.read().decode()
        remote_nics.add(stdout.strip())
        if stderr:
            print('From remote %s stderr:\n %s' % (hostname, stderr.strip()))
        ssh.close()

    remote_nics.add(local_nic)
    return list(remote_nics)


def get_subnet(local_address, remote_hostnames, identify_file=''):
    ssh_directory = os.path.expanduser('~/.ssh') if identify_file == '' else os.path.dirname(
        os.path.abspath(os.path.expanduser(identify_file)))
    config = paramiko.config.SSHConfig.from_path(
        os.path.join(ssh_directory, 'config'))
    remote_address = [config.lookup(hostname)['hostname']
                      for hostname in remote_hostnames]
    remote_address.append(local_address)
    address_pool = set()
    for addr in remote_address:
        binary_repr = int(''.join([format(int(part), '08b')
                                   for part in addr.split('.')]), 2)
        address_pool.add(format(binary_repr+1, '032b'))
        address_pool.add(format(binary_repr-1, '032b'))
    address_pool = list(address_pool)
    longestCommonPrefix = 0
    for item in zip(*address_pool):
        if len(set(item)) > 1:
            break
        longestCommonPrefix += 1
    if longestCommonPrefix > 30:
        longestCommonPrefix = 30
    assert longestCommonPrefix >= 16, 'Hosts not in the same subnet!'
    commonAddress = address_pool[0][:longestCommonPrefix] + \
        '0' * (32 - longestCommonPrefix)
    parts = [commonAddress[:8], commonAddress[8:16],
             commonAddress[16:24], commonAddress[24:]]
    subnet = '.'.join([str(int(part, 2))
                       for part in parts]) + '/%d' % longestCommonPrefix
    return subnet


def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Configuration file.')
    parser.add_argument('-i', '--identify', default='',
                        help='SSH identify file.')
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Command to be executed.')
    args = parser.parse_args()
    settings = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
    attributes = set(['host', 'servers', 'workers', 'chief'])
    hosts = []
    servers, workers = {}, {}
    chief = None
    chief_address = socket.gethostbyname(socket.gethostname())
    port = get_available_port(chief_address)
    for node in settings['nodes']:
        assert set(node.keys(
        )) <= attributes, 'Attributes of nodes invalid, %s / %s.' % (set(node.keys()), attributes)
        hosts.append(node['host'])
        if node.get('servers', 0):
            servers[node['host']] = node['servers']
        if node.get('workers', 0):
            workers[node['host']] = node['workers']
        if node.get('chief', False):
            assert chief is None, 'There should be only one chief.'
            chief = node['host']
    assert chief, 'There should be one chief.'
    num_servers = sum(servers.values())
    num_workers = sum(workers.values())
    enable_PS = (num_servers > 0)
    print('Cluster: {')
    print('  Chief: %s,' % chief)
    print('  Servers(%d): %s,' % (num_servers, servers))
    print('  Workers(%d): %s,' % (num_workers, workers))
    print('}')
    if enable_PS:
        os.environ['DMLC_PS_ROOT_URI'] = chief_address
        os.environ['DMLC_PS_ROOT_PORT'] = str(port)
        os.environ['DMLC_PS_VAN_TYPE'] = 'p3'
        os.environ['DMLC_NUM_SERVER'] = str(num_servers)
        os.environ['DMLC_NUM_WORKER'] = str(num_workers)

    global executor_shell
    if len(hosts) == 1:
        # single machine
        # TODO: add hostdress validation check
        if enable_PS:
            proc = multiprocessing.Process(target=start_sched)
            _procs.append(proc)
            for i in range(num_servers):
                proc = multiprocessing.Process(target=start_server)
                _procs.append(proc)
        for proc in _procs:
            proc.start()
        mpi_command = 'mpirun --allow-run-as-root --tag-output -np %d %s' % (
            num_workers, ' '.join(args.command))
        env = dict(os.environ)
        if enable_PS:
            env["DMLC_ROLE"] = "worker"
        executor_shell = subprocess.Popen(
            mpi_command, shell=True, env=env, stdout=None, stderr=None)
        for proc in _procs:
            proc.join()
        executor_shell.wait()
    else:
        # multi machines

        #! nic names not used currently, use subnets instead; nccl_socket_name please specified in /etc/bash.bashrc
        #! nic methods cannot support different nic name on different machines
        # nics = get_nic_names(chief_address, set(hosts) - {chief}, args.identify)
        # joined_nics = ','.join(nics)
        subnet = get_subnet(chief_address, set(hosts) - {chief}, args.identify)
        if enable_PS:
            with open('/tmp/temp_hetu_config.yml', 'w') as fw:
                yaml.dump({'shared': {'DMLC_PS_ROOT_URI': chief_address, 'DMLC_PS_ROOT_PORT': port,
                                      'DMLC_NUM_WORKER': num_workers, 'DMLC_NUM_SERVER': num_servers, 'DMLC_PS_VAN_TYPE': 'p3'}}, fw)
            proc = multiprocessing.Process(target=start_sched)
            _procs.append(proc)
        for node in hosts:
            if node == chief:
                for i in range(servers.get(node, 0)):
                    proc = multiprocessing.Process(target=start_server)
                    _procs.append(proc)
            else:
                if servers.get(node, 0):
                    proc = multiprocessing.Process(target=start_remote_server, args=[
                                                   node, servers[node], args.identify])
                    _procs.append(proc)
        for proc in _procs:
            proc.start()
        basic_args = '--allow-run-as-root --tag-output'
        hosts_in_command = ','.join(
            ['%s:%d' % (node, nworkers) for node, nworkers in workers.items()])
        mpi_ssh_args = '' if args.identify == '' else '-bootstrap=ssh -bootstrap-exec-args -i %s' % args.identify
        tcp_intf_arg = '-mca btl_tcp_if_include %s' % subnet
        # tcp_intf_arg = '-mca btl_tcp_if_include %s' % joined_nics
        # nccl_socket_intf_arg = '-x NCCL_SOCKET_IFNAME=%s' % joined_nics
        env_list = '-x DMLC_PS_ROOT_URI=%s -x DMLC_PS_ROOT_PORT=%s -x DMLC_PS_VAN_TYPE=p3 -x DMLC_NUM_SERVER=%s -x DMLC_NUM_WORKER=%s -x DMLC_ROLE=worker' %\
            (chief_address, str(port), str(num_servers),
             str(num_workers)) if enable_PS else ''
        mpi_command = (
            'mpirun {basic_args} '
            '--host {hosts} '
            '{mpi_ssh_args} '
            '{tcp_intf_arg} '
            # '{nccl_socket_intf_arg} '
            '{env} '
            '{command}'
            .format(basic_args=basic_args,
                    hosts=hosts_in_command,
                    mpi_ssh_args=mpi_ssh_args,
                    tcp_intf_arg=tcp_intf_arg,
                    # nccl_socket_intf_arg=nccl_socket_intf_arg,
                    env=env_list,
                    command=' '.join(args.command))
        )
        executor_shell = subprocess.Popen(
            mpi_command, shell=True, stdout=None, stderr=None)
        for proc in _procs:
            proc.join()
        executor_shell.wait()


if __name__ == '__main__':
    #! need to modify /etc/bash.bashrc on other machines for:
    #       * specify NCCL_SOCKET_IFNAME
    #       * specify PATH for mpirun support
    #       * activate conda environment
    #       * specify PYTHONPATH for hetu support
    #! ssh process to other machines for server CANNOT receive SIGINT from Ctrl+C on this machine, please kill on other machines
    main()
