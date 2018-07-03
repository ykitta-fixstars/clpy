import sys

if len(sys.argv)<2:
    sys.exit()

sys.argv.pop(0)
with open(sys.argv[0]) as f:
    exec(f.read())
