
RNN with Tensorflow and pyTorch in a two GPU Setup
---------------

The repo contains the code for evaluating the GPU usage in two GPU setup when running a 
model with two RNN layers (one of which using a tanh and other using relu),
both of which are then concatenated and filtered with a Dropout layer.
There four different implementations of the this simple model:

1. Using Tensorflow, with each RNN layer is keras.layer.RNN
2. Using Tensorflow, with each RNN is unrolled with a hand-written layer
3. Using pyTorch, with each RNN is torch.nn.RNN.
4. Using pyTorch, with each RRN is unrolled with a troch.nn.Module.

The first two models are in rnn_tf.py, where as the later two are in rnn_torch.py.

The four implementations are run using IMDB dataset for each of the following two cases w.r.t number of GPUs.

1. when each RNN layer is pinned to two distinct GPUs (GPU0 and GPU1), and
2. when both RNN layers are pinned the first GPU (GPU0).

In both cases, the output is pinned to the first GPU.

Results (table below) show that using two keras RNN layers causes both the layers to  be located
to the first GPU, where as the use of hand rolled layers retains the computations
in the respective GPUs.

However, pyTorch allows the use of nn.RNN module while pinning the computations on the GPU.


|Library|Layer|GPUs|Test Accuracy|Val Accuracy|Time (s)|GPU0 Avg Usage|GPU0 Peak Usage|GPU0 Avg Mem (MB) |GPU0 Peak Mem (MB)|GPU1 Avg Usage|GPU1 Peak Usage|GPU1 Avg Mem (MB)|GPU1 Peak Mem (MB)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Tensorflow|keras|1|83.14%|83.32%|125.7|7.40%|11.00%|235.6|240|0.00%|0.00%|166|166|
|Tensorflow|Manual|1|82.98%|82.24%|98.7|11.30%|13.00%|240|240|0.00%|0.00%|166|166|
|Tensorflow|keras|2|82.55%|81.56%|94.9|10.40%|12.00%|240|240|0.90%|1.00%|173.8|174|
|Tensorflow|Manual|2|82.34%|82.68%|599.8|21.70%|96.00%|240|240|16.20%|19.00%|228.9|240|
|Torch|nn.RNN|1|67.14%|67.14%|116.1|10.80%|12.00%|259.10|260.00|0.00%|0.00%|4|4|
|Torch|Manual|1|61.01%|61.01%|108.3|11.60%|13.00%|260.00|260.00|0.00%|0.00%|4|4|
|Torch|nn.RNN|2|64.80%|64.80%|89.2|7.70%|10.00%|257.50|258.00|7.20%|9.00%|256.7|258|
|Torch|Manual|2|56.44%|56.44%|84.2|8.10%|10.00%|257.50|258.00|7.80%|9.00%|257.3|258|
