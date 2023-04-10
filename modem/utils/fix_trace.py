import click
import numpy as np
import time
import os
import glob
from pathlib import Path
from robohive.logger.grouped_datasets import Trace

DESC='Fix grasp pos in existing trace'
@click.command(help=DESC)
@click.option('-td', '--trace_dir', type=str, help='absolute path to trace to load', required=True)
def main(trace_dir):
    fps = glob.glob(trace_dir+"/*.pickle")
    for f_idx,fp in enumerate(fps):
        print('Path {} of {}'.format(f_idx, len(fps)))
        paths = Trace.load(fp)
        if 'env_infos' in paths.trace['Trial0']:
            paths.close()
        for pname, pdata in paths.trace.items():
            print(pname)
            qp = pdata['env_infos/obs_dict/qp']
            qv = pdata['env_infos/obs_dict/qv']
            grasp_pos_od = pdata['env_infos/obs_dict/grasp_pos']
            grasp_pos_obs = pdata['observations'][:, qp.shape[1]+qv.shape[1]:qp.shape[1]+qv.shape[1]+3]
            assert((np.abs(grasp_pos_obs[1:] - grasp_pos_obs[0]) > 1e-5).any())
            assert((np.abs(grasp_pos_od - grasp_pos_od[0]) < 1e-5).all())
            assert(grasp_pos_obs.shape == grasp_pos_od.shape)
            paths.trace[pname]['env_infos/obs_dict/grasp_pos'] = grasp_pos_obs.copy()

            if 'env_infos/proprio_dict/grasp_pos' in pdata:
                grasp_pos_pd = pdata['env_infos/proprio_dict/grasp_pos']
                assert((np.abs(grasp_pos_pd - grasp_pos_pd[0]) < 1e-5).all())
                assert(grasp_pos_obs.shape == grasp_pos_pd.shape)
                paths.trace[pname]['env_infos/proprio_dict/grasp_pos'] = grasp_pos_obs.copy()
        paths.save(trace_name=fp, verify_length=True, f_res=np.float64)

    #paths.close()
    #output_dir = '/'.join(trace_path.split('/')[:-1])
    #paths.render(output_dir=output_dir, output_format="mp4", groups=":", datasets=RENDER_KEYS, input_fps=FPS)

if __name__ == '__main__':
    main()
