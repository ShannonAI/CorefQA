#!/usr/bin/env python3 
# -*- coding: utf-8 -*-  



# author: xiaoy li 
# description:
# test config in model fn builder 




def model_fn(config):

    def mention_proposal_fn():
        print("the number of document is : ")
        print(config.document_number)
        print(config.number_window_size)

    return mention_proposal_fn 


class Config:
    number_window_size = 2 
    document_number = 3 




if __name__ == "__main__":
    config = Config()
    get_model_fn = model_fn(config)
    get_model_fn()