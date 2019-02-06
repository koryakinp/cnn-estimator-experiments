import tensorflow as tf
import numpy as np
from lib.mnist import MNIST
from estimator import *
from utils import *
from network import get_model

data = MNIST(data_dir="data/MNIST/")

estimator = Estimator(data, "./models", 128)

estimator.build_model()
