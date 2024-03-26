import torch
import numpy as np

from pyhessian.utils import (
    group_product,
    group_add,
    normalization,
    get_params_grad,
    hessian_vector_product,
    orthnormal,
)


class Hessian:
    def __init__(
        self, model, loss_fn, data=None, dataloader=None, max_iter=100, cuda=True
    ):
        assert (data is not None and dataloader is None) or (
            data is None and dataloader is not None
        )
        self.model = model.eval()
        self.criterion = loss_fn
        self.max_iter = max_iter

        if data is not None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        self.device = "cuda" if cuda else "cpu"

        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == "cuda":
                self.inputs, self.targets = self.inputs.cuda(), self.targets.cuda()
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        self.params, self.gradsH = get_params_grad(self.model)

    def dataloader_hv_product(self, v):
        device = self.device
        num_data = 0
        THv = [torch.zeros(p.size()).to(device) for p in self.params]

        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(
                gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False
            )
            THv = [THv1 + Hv1 * float(tmp_num_data) for THv1, Hv1 in zip(THv, Hv)]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, max_iter=100, tol=1e-3, top_n=1):
        assert top_n >= 1

        device = self.device
        eigenvalues = []
        eigenvectors = []
        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params]
            v = normalization(v)

            for _ in range(max_iter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if (
                        abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                        < tol
                    ):
                        break
                    eigenvalue = tmp_eigenvalue

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, max_iter=100, tol=1e-3):
        device = self.device
        trace_vhv = []
        trace = 0.0

        for _ in range(max_iter):
            self.model.zero_grad()
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iterations=100, num_runs=1):
        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for _ in range(num_runs):
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []

            for i in range(iterations):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.0:
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iterations, iterations).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.linalg.eig(T)

            eigen_list = a_
            weight_list = torch.pow(b_, 2)
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full
