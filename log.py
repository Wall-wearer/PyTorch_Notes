# This code is used to active TensorBoard and name it as 'logs'. Everytime I use this code, there will be a log file
# created under the logs folder.

from tensorboardX import SummaryWriter

writer = SummaryWriter('./logs')

