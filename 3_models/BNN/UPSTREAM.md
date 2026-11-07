# BinaryNet provenance

The deployed BNN is the binary LeNet-5 architecture and pretrained MNIST
checkpoint from RWTH-IDS/efficient-binary-neural-networks (MIT license):

https://github.com/RWTH-IDS/efficient-binary-neural-networks

The graph is adapted only to expose the platform's deployment and training
interfaces. Its layer order, binary operations, batch-normalization state and
`[0, 1]` input preprocessing are preserved.

Upstream reported accuracy: 99.16%. Platform full-test verification: 99.19%
(9,919 / 10,000 samples).
