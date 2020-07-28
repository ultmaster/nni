import random

from nni.tuner import Tuner


random.seed(0)


class NaiveRandomTuner(Tuner):
    def update_search_space(self, search_space):
        self.search_space = search_space
    
    def generate_parameters(self, parameter_id, **kwargs):
        ret = {}
        for k, domain in self.search_space.items():
            assert domain['_type'] == 'choice'
            ret[k] = random.choice(domain['_value'])
        return ret

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        pass
