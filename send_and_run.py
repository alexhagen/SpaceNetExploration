import sys
import os
import paramiko
cmd = 'sshpass -f /Users/hage581/.ohmahgerd rsync -v -r -a -P --exclude="img/" --exclude="checkpoints/" -e ssh ./ hage581@ohmahgerd.pnl.gov:/qfs/projects/sgdatasc/SpaceNetExploration'
result = os.popen(cmd).read()
print(result)

#client = paramiko.client.SSHClient()
#client.load_system_host_keys()
#with open('/Users/hage581/.ohmahgerd', 'r') as f:
#    p = f.readline().strip()
#client.connect('ohmahgerd.pnl.gov', username='hage581', password=p)
#stdin, stdout, stderr = client.exec_command('cd /qfs/projects/sgdatasc/SpaceNetExploration')
#stdin, stdout, stderr = client.exec_command('python3.6 training/train.py')
#print(stdout.read())
#client.close()
