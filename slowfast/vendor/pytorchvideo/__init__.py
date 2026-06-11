# NOTICE
#
# The modules in this package are vendored from
# https://github.com/facebookresearch/pytorchvideo (Apache License 2.0),
# which is no longer actively maintained and is incompatible with recent
# torchvision releases. Only the small, pure-PyTorch pieces that slowfast
# depends on are vendored here:
#
#   distributed.py                 <- pytorchvideo/layers/distributed.py
#   batch_norm.py                  <- pytorchvideo/layers/batch_norm.py
#   swish.py                       <- pytorchvideo/layers/swish.py
#   soft_target_cross_entropy.py   <- pytorchvideo/losses/soft_target_cross_entropy.py
#   utils.py                       <- pytorchvideo/layers/utils.py
#   functional.py                  <- convert_to_one_hot from pytorchvideo/transforms/functional.py
#
# Original copyright headers are retained in each file. Local changes are
# limited to import rewrites (absolute pytorchvideo imports -> relative) and
# replacing a private torch._C import with its public torch.distributed alias.
