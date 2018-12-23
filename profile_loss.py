# wujian@2018

import time
import torch as th
import torch.nn.functional as F


def ge2e_v1(e, N, M):
    """
    e: N x M x D, after L2 norm
    N: number of spks
    M: number of utts
    """
    # N x D
    c = th.mean(e, dim=1)
    s = th.sum(e, dim=1)
    # build similarity matrix
    dst = []
    # jth speaker
    for j in range(N):
        # ith utterance
        for i in range(M):
            # kth ref speaker
            for k in range(N):
                if k == j:
                    # fix centroid
                    cj = (s[j] - e[j][i]) / (M - 1)
                    dst.append(th.dot(e[j][i], cj))
                else:
                    dst.append(th.dot(e[j][i], c[k]))
    # N*M*N
    sim = th.stack(dst)
    # N*M x N
    sim = sim.view(-1, N)
    # build label N*M
    ref = th.zeros(N * M, dtype=th.int64, device=e.device)
    for r, s in enumerate(range(0, N * M, M)):
        ref[s:s + M] = r
    # ce loss
    loss = F.cross_entropy(sim, ref)
    return loss


def ge2e_v2(e, N, M):
    """
    e: N x M x D, after L2 norm
    N: number of spks
    M: number of utts
    """
    # N x D
    c = th.mean(e, dim=1)
    s = th.sum(e, dim=1)
    # build similarity matrix
    # NM * D
    e = e.view(N * M, -1)
    # NM * N
    sim = th.mm(e, th.transpose(c, 0, 1))
    # fix similarity matrix
    for j in range(N):
        for i in range(M):
            cj = (s[j] - e[j*M + i]) / (M - 1)
            sim[j*M + i][j] = th.dot(cj, e[j*M + i])
    # build label N*M
    ref = th.zeros(N * M, dtype=th.int64, device=e.device)
    for r, s in enumerate(range(0, N * M, M)):
        ref[s:s + M] = r
    # ce loss
    loss = F.cross_entropy(sim, ref)
    return loss


def foo():
    N, M, D = 64, 20, 64
    e = th.rand(N, M, D)
    e = e / th.norm(e, dim=-1, keepdim=True)
    s = time.time()
    loss = ge2e_v1(e, N, M)
    t = time.time()
    print(loss.data)
    print("cost: {:.2f}".format(t - s))
    s = time.time()
    loss = ge2e_v2(e, N, M)
    t = time.time()
    print(loss.data)
    print("cost: {:.2f}".format(t - s))

if __name__ == "__main__":
    foo()