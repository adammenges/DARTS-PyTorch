import  torch
import  numpy as np
from    torch import optim, autograd


def concat(xs):
    """
    flatten all tensor from [d1,d2,...dn] to [d]
    and then concat all [d_1] to [d_1+d_2+d_3+...]
    :param xs:
    :return:
    """
    return torch.cat([x.view(-1) for x in xs])




class Arch:

    def __init__(self, model, args):
        """

        :param model: network
        :param args:
        """
        self.momentum = args.momentum
        self.wd = args.wd
        self.model = model
        # this is the optimizer to optimize alpha parameter
        self.optimizer = optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_lr,
                                          betas=(0.5, 0.999),
                                          weight_decay=args.arch_wd)

    def comp_unrolled_model(self, x, target, eta, optimizer):
        """

        :param x:
        :param target:
        :param eta:
        :param optimizer: optimizer of theta, not optimizer of alpha
        :return:
        """
        # forward to get loss
        loss = self.model.loss(x, target)
        # flatten current weights
        theta = concat(self.model.parameters()).data
        # theta: torch.Size([1930618])
        # print('theta:', theta.shape)
        try:
            moment = concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)
        except:
            moment = torch.zeros_like(theta)

        # flatten all gradients
        dtheta = concat(autograd.grad(loss, self.model.parameters())).data

        dtheta = dtheta + self.wd * theta
        # indeed, here we implement a simple SGD with momentum and weight decay
        # theta = theta - eta * (moment + dtheta)
        theta = theta.sub(eta, moment + dtheta)
        # construct a new model
        unrolled_model = self.construct_model_from_theta(theta)

        return unrolled_model

    def step(self, x_train, target_train, x_valid, target_valid, eta, optimizer, unrolled):
        """

        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta:
        :param optimizer: theta optimizer
        :param unrolled:
        :return:
        """
        # alpha optimizer
        self.optimizer.zero_grad()

        if unrolled:
            self.backward_step_unrolled(x_train, target_train, x_valid, target_valid, eta, optimizer)
        else:
            self.backward_step(x_valid, target_valid)

        self.optimizer.step()

    def backward_step(self, x_valid, target_valid):
        loss = self.model.loss(x_valid, target_valid)
        loss.backward()

    def backward_step_unrolled(self, x_train, target_train, x_valid, target_valid, eta, optimizer):
        """

        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta:
        :param optimizer:
        :return:
        """
        unrolled_model = self.comp_unrolled_model(x_train, target_train, eta, optimizer)
        unrolled_loss = unrolled_model.loss(x_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, x_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            # g = g - eta * ig
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def construct_model_from_theta(self, theta):
        """
        construct a new model with initialized weight from theta
        :param theta: flatten weights, need to reshape to original shape
        :return:
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = v.numel()
            # restore theta[] value to original shape
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model.loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model.loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
