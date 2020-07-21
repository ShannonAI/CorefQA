#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# config utils for the mention proposal and corefqa 



import os 
import json 
import tensorflow as tf 


class ModelConfig(object):
    def __init__(self, tf_flags, output_dir, model_sign="model"):
        key_value_pairs = tf_flags.flag_values_dict()

        for item_key, item_value in key_value_pairs.items():
            self.__dict__[item_key] = item_value 

        self.output_dir = output_dir 
        config_path = os.path.join(self.output_dir, "{}_config.json".format(model_sign))
        with open(config_path, "w") as f:
            json.dump(self.__dict__, f, sort_keys=True, indent=2, ensure_ascii=False)

    def logging_configs(self):
        tf.logging.info("$*$"*30)
        tf.logging.info("****** print model configs : ******")
        tf.logging.info("$*$"*30)

        for item_key, item_value in self.__dict__.items():
            tf.logging.info("{} : {}".format(str(item_key), str(item_value)))
