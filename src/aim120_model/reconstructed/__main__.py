"""Run the same default public engine from a self-contained installed package."""
import argparse
import json
from pathlib import Path
from . import load_profile
from .resources import read
from ..public_api import simulate


def main():
    parser=argparse.ArgumentParser(description='Reconstructed missile candidate: IR / ARH / explicit PD-SARH')
    parser.add_argument('--missile',default='su_r_77_1')
    parser.add_argument('--input',type=Path,help='Independent scene JSON; package example if omitted')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    scenario=json.loads(args.input.read_text(encoding='utf-8')) if args.input else read('example_scene.json')
    result=simulate(load_profile(args.missile),scenario)
    if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],ensure_ascii=False,allow_nan=False,indent=2))


if __name__=='__main__':main()
