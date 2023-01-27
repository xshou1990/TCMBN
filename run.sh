device=0
data="data/defi/split_1/"
batch=8
n_head=1
n_layers=1
d_model=32
d_inner=16
d_k=16
d_v=16
ber_comps=64
gau_comps=32
dropout=0.1
lr=6e-3
epoch=100
log=log.txt


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -ber_comps $ber_comps -dropout $dropout -lr $lr -epoch $epoch -log $log


# defi
# device=0
# data="data/defi/split_1/"
# batch=8
# n_head=1
# n_layers=1
# d_model=32
# d_inner=16
# d_k=16
# d_v=16
# ber_comps=64
# gau_comps=32
# dropout=0.1
# lr=6e-3
# epoch=100
# log=log.txt

# synthea
# device=0
# data="data/synthea/split_1/"
# batch=8
# n_head=4
# n_layers=1
# d_model=64
# d_inner=32
# d_k=32
# d_v=32
# ber_comps=64
# gau_comps=64
# dropout=0.1
# lr=2e-3
# epoch=100
# log=log.txt

# mimic
# device=0
# data="data/mimic3/split_1/"
# batch=16
# n_head=1
# n_layers=1
# d_model=64
# d_inner=32
# d_k=32
# d_v=32
# ber_comps=64
# gau_comps=64
# dropout=0.1
# lr=0.0015
# epoch=100
# log=log.txt

#dunnhumby
# device=0
# data="data/dunnhumby/split_1/"
# batch=8
# n_head=6
# n_layers=1
# d_model=32
# d_inner=16
# d_k=16
# d_v=16
# ber_comps=64
# gau_comps=64
# dropout=0.1
# lr=0.01
# epoch=100
# log=log.txt


# poisson_mbn, hawkes_mbn
# device=0
# data="data/poisson_mbn/split_1/"
# batch=8
# n_head=4
# n_layers=1
# d_model=64
# d_inner=32
# d_k=32
# d_v=32
# ber_comps=64
# gau_comps=64
# dropout=0.1
# lr=2e-4
# epoch=100
# log=log.txt

