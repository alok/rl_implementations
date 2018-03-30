PyTorch Implementation of OpenAI's
[REPTILE](https://blog.openai.com/reptile/) algorithm. Slightly longer
but a bit cleaner than John's
[gist](https://gist.github.com/joschu/f503500cda64f2ce87c8288906b09e2d#file-reptile-sinewaves-demo-py).

It trains 30,000 meta-iterations of gradient descent, one task at a
time. It evaluates on a fixed task every 1,000 iterations, taking 5
gradient descent steps per task. Turns out that 1 is enough to get good
performance, indicating that meta-learning is actually working.

## Requirements

-   Python 3.6+
-   Numpy
-   Matplotlib

## Running the Script

    python3 main.py

It will pop up a Matplotlib window that updates every 3,000 iterations.
