# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from pytorch_lightning import LightningModule
import torch
from tape.models.modeling_utils import ProteinConfig
from tape.models.modeling_bert import ProteinBertAbstractModel, ProteinBertModel
from importlib import import_module
import numpy as np

def load_featmodel_from_checkpoint(model_config_file,feat_path,strict_reload=False):
    #module = getattr(import_module("src.models.feat_extraction"), "FeatExtractor")
    #print(module)
    feat_model = FeatExtractor.load_from_checkpoint(feat_path, model_config_file=model_config_file, strict=strict_reload)
    feat_model.eval()
    return feat_model

class BertFeatureExtractor(ProteinBertAbstractModel):
    """BertFeatureExtractor as tape model"""

    def __init__(self, config):
        super().__init__(config)
        self.bert = ProteinBertModel(config)
        self.hidden_size_bert = config.hidden_size


    def forward(self, input_ids, variable_regions, input_mask=None, targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]
        if None not in variable_regions: # extract the variable regions instead of the first token
            Output = torch.empty(pooled_output.shape)
            for i, vr in enumerate(variable_regions):
                vr_ = [x for x in vr if not np.isnan(x)]
                Output[i] = sequence_output[i, vr_, :].mean(0)
        else:
             Output = pooled_output

        if torch.cuda.is_available():
            Output = Output.cuda()

        return Output


class FeatExtractor(LightningModule):
    """BertFeatureExtractor as lightning model"""
    
    def __init__(self, model_config_file):
        super().__init__()
        config = ProteinConfig().from_pretrained(model_config_file)
        self.model = BertFeatureExtractor(config)
    
    def forward(self, input_ids, input_mask=None, variable_regions=None, targets=None):
        return self.model(input_ids, input_mask=input_mask, targets=targets, variable_regions=variable_regions)
