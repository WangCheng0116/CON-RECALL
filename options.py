# This implementation is adapted from ReCall: https://github.com/ruoyuxie/recall
import argparse
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="EleutherAI/pythia-6.9b", help="the model to attack")
        self.parser.add_argument('--ref_model', type=str, default="EleutherAI/pythia-160m")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument('--dataset', type=str, help="dataset name")
        self.parser.add_argument('--sub_dataset', type=str, default=128, help="")
        self.parser.add_argument('--num_shots', type=str, default="12", help="number of shots to evaluate.")
        self.parser.add_argument('--pass_window', type=bool, default=True, help="whether to pass the window to the model.")        
        self.parser.add_argument('--attack_type', type=str, choices=['del', 'sub', 'para', 'none'], default='none', 
                                 help="Type of attack to apply (del, sub, para, or none)")
        self.parser.add_argument('--attack_strength', type=float, default=0.0, 
                                 help="Strength of the attack (portion of words to modify, between 0 and 1)")