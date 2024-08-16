import torch
from models.optimus_5prime.model import MeanRibosomeLoadModule
from models.optimus_5prime.data import *
from seqopt.oracle import NucleotideDesigner
from trials import parse_args, setup_model_from_lightning, lower_confidence_bound,run_all_trials,tune_langevin
from functools import partial

class MRLDesigner(NucleotideDesigner):
        
        def __init__(self,forward_fn):
            super().__init__(num_classes=12,class_dim=2)
            self.fwd = forward_fn
        
        def onehot_encode(self,seq):
            '''Index tensor to one-hot encoding'''
            utr = torch.nn.functional.one_hot(seq,num_classes=self.num_classes).float().requires_grad_(True)
            return utr
    
        def dense_decode(self,seq):
            ''' A method to convert a dense sequence to a readable nucleotide sequence'''
            print(seq) 
            if seq.dim() > 1:
                seq = seq[0,:]
            as_list = seq.cpu().numpy().ravel().tolist()
            vocab = {'[CLS]': 0, '[SEP]': 1, '[BOS]': 2, '[MASK]': 3, '[PAD]': 4, '[RESERVED]': 5, '[UNK]': 6, 'A': 7, 'C': 8, 'G': 9, 'T': 10, 'N': 11} 
            #mapping = 'ACGT'
            mapping = { v : k for k,v in vocab.items()}
            nucs = [mapping[x] for x in as_list]
            return ''.join(nucs)
    
        def oracles(self):
            ''' A method to evaluate a sequence'''
            return [self.fwd]
    
        def seed_sequence(self):
            '''Generate a random sequence of a given length'''
            return torch.randint(7,11,(1,50)) 

if __name__ == "__main__":

    args = parse_args()
    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    module = MeanRibosomeLoadModule.load_from_checkpoint(args.checkpoint,
                                                         map_location=device)
    
    # extact PyTorch model and wrap 
    mrl = setup_model_from_lightning(module,device)
    if mrl.evidential:
        lcb = partial(lower_confidence_bound,mrl)
        designer = MRLDesigner(lcb)
    else: 
        designer = MRLDesigner(mrl)

    # all PR and MCMC trials
    run_all_trials(designer,args)

