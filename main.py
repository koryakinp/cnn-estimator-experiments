import tensorflow as tf
import numpy as np
from lib.mnist import MNIST
from estimator import *
from utils import *
from network import get_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

data = MNIST(data_dir="data/MNIST/")

estimator = Estimator(data, "./models", 128)

estimator.build_model()
