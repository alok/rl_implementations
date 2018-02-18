This repo contains code to solve the one of the warm-ups for the [OpenAI
Requests for Research
2.0](https://blog.openai.com/requests-for-research-2/).

It requires Python 3.6+, Numpy, and PyTorch. It's been tested with a
GPU, but should work without one.

Usually about 5-10,000 training samples are enough. It trains online,
with a batch size of 1, though that can be changed.

It will accept variable length strings.
