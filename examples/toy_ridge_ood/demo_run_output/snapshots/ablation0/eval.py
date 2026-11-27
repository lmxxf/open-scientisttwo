import argparse, json
import numpy as np
from data import data
from model import fit_predict

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['subset', 'full'], required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--config', default='config.json')
    p.add_argument('--extra', action='store_true')
    args = p.parse_args()
    config = json.load(open(args.config))
    seeds = list(range(10, 15)) if args.extra else list(range(5))
    values = []
    for seed in seeds:
        x, y = data(seed, 80 if args.mode == 'subset' else 400, 'train')
        t, target = data(seed, 300, args.split)
        pred = fit_predict(x, y, t, config)
        if pred.shape != target.shape or not np.isfinite(pred).all():
            raise ValueError('invalid prediction')
        values.append(float(np.mean((pred-target)**2)))
    print(json.dumps(dict(mse=float(np.mean(values)), per_seed=values, seeds=seeds,
                          mode=args.mode, split=args.split, extra=args.extra)))
