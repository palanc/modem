import os
import torch
from copy import deepcopy
from algorithm.tdmpc import TDMPC
from omegaconf import OmegaConf,open_dict

ROOT_DIR = '/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d'
IN_DIR = ROOT_DIR+'/bin_reorient_real_bc_100demos'
OUT_DIR = ROOT_DIR+'/bin_reorient_ensemble_bc_100demos'
IN_NAME = 'bc_seed4.pt'
OUT_NAME = 'bc_seed4.pt'
ENSEMBLE_SIZE = 6

OLD_AGENT_FP = IN_DIR+'/'+IN_NAME
NEW_AGENT_FP = OUT_DIR+'/'+OUT_NAME

def convert_agent():

    # Make sure root dir exists
    if not os.path.isdir(ROOT_DIR):
        print('Root dir {} does not exist, exiting'.format(ROOT_DIR))
        exit()

    # Make sure old agent exists
    if not os.path.exists(OLD_AGENT_FP):
        print('Did not find old agent at {}, exiting'.format(OLD_AGENT_FP))
        exit()

    # Make sure new agent doesn't already exist
    if os.path.exists(NEW_AGENT_FP):
        print('Agent at {} already exists, exiting'.format(NEW_AGENT_FP))
        exit()

    # Create new directory if necessary
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    # Create new agent
    old_agent_data = torch.load(OLD_AGENT_FP)

    new_cfg = deepcopy(old_agent_data.cfg)
    OmegaConf.set_struct(new_cfg, True)
    with open_dict(new_cfg):
        new_cfg.ensemble_size = ENSEMBLE_SIZE
    new_agent = TDMPC(new_cfg)

    new_agent.model._encoder.load_state_dict(old_agent_data.model._encoder.state_dict())
    new_agent.model._state_encoder.load_state_dict(old_agent_data.model._state_encoder.state_dict())

    new_agent.model._learn_encoder.load_state_dict(old_agent_data.model._encoder.state_dict())
    new_agent.model._learn_state_encoder.load_state_dict(old_agent_data.model._state_encoder.state_dict())

    new_agent.model_target._encoder.load_state_dict(old_agent_data.model._encoder.state_dict())
    new_agent.model_target._state_encoder.load_state_dict(old_agent_data.model._state_encoder.state_dict())

    new_agent.model_target._learn_encoder.load_state_dict(old_agent_data.model._encoder.state_dict())
    new_agent.model_target._learn_state_encoder.load_state_dict(old_agent_data.model._state_encoder.state_dict())

    for i in range(0,ENSEMBLE_SIZE):
        new_agent.model._acs[i]._pi.load_state_dict(old_agent_data.model._pi.state_dict())
        new_agent.model_target._acs[i]._pi.load_state_dict(old_agent_data.model._pi.state_dict())

    torch.save(new_agent, NEW_AGENT_FP)
    print('Saved new agent at {}'.format(NEW_AGENT_FP))

if __name__ == '__main__':
    convert_agent()