import argparse

p = argparse.ArgumentParser(description='Play with models')
p.add_argument('mode', action='store')
args = p.parse_args()
