# Define special symbols and indices used in seq2seq models.
# NOTE: Not all models use these indexes; they are meant for use in seq2seq
# models and cannot in general be relied upon, as many other models, such as
# multi-head models, do not use this convention and indeed often have no notion
# of unknown, padding, BoSs, EoSs, or seperators.
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX = 0, 1, 2, 3, 4
