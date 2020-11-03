import json
import os
import sys
import threading
import unittest

from nni.retiarii import Model, submit_models
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.integration import RetiariiAdvisor, register_advisor
from nni.retiarii.trainer import PyTorchImageClassificationTrainer
from nni.retiarii.utils import import_


class CodeGenTest(unittest.TestCase):
    def test_mnist_example_pytorch(self):
        with open('mnist_pytorch.json') as f:
            model = Model._load(json.load(f))
            script = model_to_pytorch_script(model)
        with open('debug_mnist_pytorch.py') as f:
            reference_script = f.read()
        self.assertEqual(script.strip(), reference_script.strip())


class TrainerTest(unittest.TestCase):
    def test_trainer(self):
        Model = import_('debug_mnist_pytorch._model')
        trainer = PyTorchImageClassificationTrainer(
            Model(),
            dataset_kwargs={'root': 'data/mnist', 'download': True},
            dataloader_kwargs={'batch_size': 32},
            optimizer_kwargs={'lr': 1e-3},
            trainer_kwargs={'max_epochs': 1}
        )
        trainer.fit()


class EngineTest(unittest.TestCase):

    def test_submit_models(self):
        os.makedirs('generated', exist_ok=True)
        from nni import protocol
        protocol._out_file = open('generated/debug_protocol_out_file.py', 'wb')
        anything = lambda: None
        advisor = RetiariiAdvisor(anything)
        with open('mnist_pytorch.json') as f:
            model = Model._load(json.load(f))
        submit_models(model, model)

        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()

    def test_execution_engine(self):
        pass
