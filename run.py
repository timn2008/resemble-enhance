import sys
nArgsMin = 3
assert len(sys.argv)>=nArgsMin

import torch
import torchaudio
w1, s1 = torchaudio.load(sys.argv[1]) # '/home/tniko/Code/Shell/GTC-2024-Mar/X.mp3'
M = torch.load('mp_rank_00_model_states.pt' if len(sys.argv)==nArgsMin else sys.argv[nArgsMin])

import resemble_enhance

import resemble_enhance.denoiser
import resemble_enhance.denoiser.denoiser
#import resemble_enhance.inference
####hp = resemble_enhance.hparams.HParams()
#import resemble_enhance.denoiser.inference
#model = resemble_enhance.denoiser.denoiser.Denoiser(hp)


# import pathlib
# import omegaconf
# H = omegaconf.OmegaConf.load('hparams.yaml')
# H.denoiser_run_dir = pathlib.Path('.')
# H.cfm_solver_method='midpoint'
# resemble_enhance.enhancer.enhancer.Enhancer(H)


import resemble_enhance.enhancer.hparams
hp = resemble_enhance.enhancer.hparams.HParams()


#hp.cfm_solver_method = "midpoint"
hp.cfm_solver_nfe = 128

# hp = H
# hp = omegaconf.OmegaConf.merge(H, hp)
####hp = resemble_enhance.denoiser.hparams.HParams()


import resemble_enhance.enhancer.enhancer

E = resemble_enhance.enhancer.enhancer.Enhancer(hp)
E.load_state_dict(M['module'])
E.eval()



# Shortest way:
res = resemble_enhance.inference.inference(E.denoiser , w1[0], s1, 'cpu')
torchaudio.save(sys.argv[2], res[0].unsqueeze(0), sample_rate=res[1])

#import sys
#for X in sys.modules.keys():
#    print(X)
