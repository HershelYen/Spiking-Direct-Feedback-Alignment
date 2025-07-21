import torch

def sparse_batch_collate(batch):
    xs, ys = zip(*batch)
    batch_size = len(xs)
    indices = []
    values = []
    device = xs[0].device
    for batch_idx, x in enumerate(xs):
        if not x.is_sparse:
            x = x.to_sparse_coo()
        idx = x.coalesce().indices()  # shape = [2, nnz]
        val = x.coalesce().values()
        # 补齐 batch 维：[batch_idx; t; n]，最后shape[3, nnz]
        batch_row = torch.full((1, idx.size(1)), batch_idx, dtype=idx.dtype, device=idx.device)
        full_indices = torch.cat([batch_row, idx], dim=0)
        indices.append(full_indices)
        values.append(val)
    cat_indices = torch.cat(indices, dim=1)
    cat_values  = torch.cat(values)
    out_shape = (batch_size, xs[0].size(0), xs[0].size(1))
    X_batch = torch.sparse_coo_tensor(cat_indices, cat_values, size=out_shape, dtype=xs[0].dtype, device=device)
    Y_batch = torch.stack(ys)
    return X_batch, Y_batch
