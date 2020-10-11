import numpy as np
import torch

class ConstraintsError(Exception):

    def __init__(self,ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo


class MetaOptim:

    def __init__(self, f, parameters, constraints,
                sigma, gamma, Epoches, epoches, lr, display, print_cycle,
                record, save_path, is_plot, plot_cycle, fast_mode):
        # essential variables
        self.cons = constraints
        self.f = f
        self.parameters = parameters
        # super parameters
        self.sigma = sigma  # penalty factor
        self.gamma = gamma  # magnifier
        self.Epoches = Epoches  # epoches for the outer loop
        self.epoches = epoches  # epoches for the inner loop (Unconstrained)
        self.lr = lr
        self.display = display  # whether print the info during optimization or not
        self.objective_plot_ls = torch.zeros((Epoches, int(epoches / plot_cycle))) # save the objective value of each iteration
        self.print_cycle = print_cycle
        self.record = record    # whether save the result or not
        self.save_path = save_path  # save the result to 'save_path' if self.record is True
        self.is_plot = is_plot  # dynamically plot the optimization process
        self.plot_cycle = plot_cycle
        self.fast_mode = fast_mode


        # check constraints
        self.check_constraints()

    def check_constraints(self):
        error_info = 'Check if your Constraints functions return column vectors.' + \
        'You can try to add \'.reshape(-1,1)\' to your constraints expression ends.'

        if 'eq' not in self.cons.keys():
            self.cons['eq'] = [lambda X: X[0].T.mm(torch.zeros_like(X[0]))]
        if 'ieq' not in self.cons.keys():
            self.cons['ieq'] = [lambda X: X[0].T.mm(torch.zeros_like(X[0]))]

        for con in self.cons['eq'] + self.cons['ieq']:
            dimen = con(self.parameters).shape
            try:
                if dimen[1] < 1:
                    raise ConstraintsError(error_info)
            except:
                raise ConstraintsError(error_info)

    def makeup_online_cons(self):
        if self.fast_mode:
            return self.cons['eq'][0](self.parameters).reshape(-1, 1), self.cons['ieq'][0](self.parameters).reshape(-1, 1)
        eq_cons = torch.tensor([])
        for eq in self.cons['eq']:
            eq_cons = torch.cat([eq_cons, eq(self.parameters)], dim=0)
        ieq_cons = torch.tensor([])
        for ieq in self.cons['ieq']:
            ieq_cons = torch.cat([ieq_cons, ieq(self.parameters)], dim=0)
        return eq_cons.reshape(-1,1), ieq_cons.reshape(-1,1)

    def print_process(self, i, epoch, objective, penalty):
        print('* Epoch %-2d, epoch %-5d' % (epoch + 1, i + self.print_cycle),
                'penalty: %-8.2f' % penalty, ', objective value: %-15.2f' % objective.item())

    def save_result(self):
        if self.record:
            for i, param in enumerate(self.parameters):
                np.savetxt(self.save_path + 'p%d.csv' % i, param.detach().numpy())

    def plot(self):
        if self.is_plot:
            import matplotlib.pyplot as plt
            ls = self.objective_plot_ls.reshape(1, -1).squeeze()
            plt.plot(range(ls.shape[0]), ls, markersize=10)
            plt.savefig('plot_%s.png' % self.__class__.__name__)


class EPFM(MetaOptim):
    """
    [Parameters]
    f (function):
        objective function
    parameters (list):
        a list whose members are your parameters requires gradient
    constraints (type: dict, it's the most important parameter):
        {'eq': [...], 'ieq': [...]}
    sigma (int):
        the penalty factor
    display (bool):
        print the process if True
    record (bool):
        save the result if True
    is_plot (bool):
        save the plot if True

    [Notes]
    1. If the initial value of vector requiring grad is very strange, the result may be strange.
    2. Every outer 'Epoch' changes based on that the inner 'epoch' reaches the optimal value.
       So Try to augment 'lr'/'epoches' if the result is strange.
    3. Make sure that your var in 'self.parameters' be engaged into computation,
       otherwise the Error ['NoneType' object has no attribute 'dim'] will be raised.
    """

    def __init__(self, f, parameters, constraints, sigma=1e3, gamma=1.5, Epoches=2, epoches=30000, lr=1e-3,
                display=True, print_cycle=1000, record=False, save_path='', is_plot=False, plot_cycle=100, fast_mode=False):
        super().__init__(f, parameters, constraints, sigma, gamma, Epoches, epoches, lr,
                    display, print_cycle, record, save_path, is_plot, plot_cycle, fast_mode)


    def optimize(self):
        def penalize():
            ieq_loss = 0
            for ieq in self.cons['ieq']:
                ieq_consval = ieq(self.parameters)
                dimen = ieq_consval.shape
                tmp = torch.pow(torch.max(-ieq_consval, torch.zeros((dimen))), 2)
                ieq_loss += torch.sum(tmp)
            eq_loss = 0
            for eq in self.cons['eq']:
                eq_consval = eq(self.parameters)
                dimen = eq_consval.shape
                tmp = torch.pow(eq_consval, 2)
                eq_loss += torch.sum(tmp)
            return ieq_loss + eq_loss

        # Optimization
        for epoch in range(self.Epoches):
            # Optimize the unconstrained problem
            opter = torch.optim.Adam(self.parameters)
            for i in range(self.epoches):
                opter.zero_grad()
                objective = self.f(self.parameters).reshape(1,1)
                # add the penalty
                eq_consval, ieq_consval = self.makeup_online_cons()
                iteral_penalty = torch.sum(torch.pow(torch.max(-ieq_consval, torch.zeros(ieq_consval.shape)), 2)) + \
                    torch.sum(torch.pow(eq_consval, 2))
                # compute the generalized objective value
                t = objective + iteral_penalty * self.sigma
                t.backward()
                opter.step()
                # print the process
                if i % self.print_cycle == 0 and self.display:
                    self.print_process(i, epoch, objective, iteral_penalty)
                if self.is_plot and i % self.plot_cycle == 0:
                    self.objective_plot_ls[epoch, int(i / self.plot_cycle)] = objective.item()

            # update the penalty factor
            self.sigma *= self.gamma

        # post process
        # save the result
        self.save_result()
        # is_plot
        #self.plot()



class ALM(MetaOptim):

    def __init__(self, f, parameters, constraints, sigma=1e3, gamma=1.5, Epoches=2, epoches=30000, lr=1e-3,
                display=True, print_cycle=1000, record=False, save_path='', is_plot=False, plot_cycle=100, fast_mode=False):
        super().__init__(f, parameters, constraints, sigma, gamma, Epoches, epoches, lr,
                    display, print_cycle, record, save_path, is_plot, plot_cycle, fast_mode)

    def optimize(self):
        # Judgement function of termination
        c = lambda eq, ieq: torch.norm(eq) + torch.norm(torch.max(torch.zeros_like(ieq), -ieq))

        # Optimization
        # Initialize lagrangian multiplier w for eq and v for ieq
        mul = lambda x: x[0] * x[1]
        w = torch.ones((sum(map(lambda x: mul(x(self.parameters).shape), self.cons['eq'])), 1))
        v = torch.ones((sum(map(lambda x: mul(x(self.parameters).shape), self.cons['ieq'])), 1))

        for epoch in range(self.Epoches):
            # Optimize the unconstrained problem
            opter = torch.optim.Adam(self.parameters)
            _eq_cons, _ieq_cons = self.makeup_online_cons()
            c_pre = c(_eq_cons, _ieq_cons)
            for i in range(self.epoches):
                # main iteration
                opter.zero_grad()
                objective = self.f(self.parameters).reshape(1,1)
                t = objective.clone()
                # Construct the augmented lagrangian objective function
                eq_cons, ieq_cons = self.makeup_online_cons()
                objective += -w.T.mm(eq_cons) + (torch.sum(torch.pow(eq_cons, 2)) * self.sigma / 2)
                objective += torch.sum((torch.pow(torch.max(v - self.sigma * ieq_cons, torch.zeros_like(v)), 2) - torch.pow(v, 2)) / (2 * self.sigma))
                objective.backward(retain_graph=True)
                opter.step()

                # print the process
                if self.display and i % self.print_cycle == 0:
                    self.print_process(i, epoch, t, c_pre)
                if self.is_plot and i % self.plot_cycle == 0:
                    self.objective_plot_ls[epoch, int(i / self.plot_cycle)] = t

            c_post = c(eq_cons, ieq_cons)
            # magnify the penalty factor
            if c_post / c_pre > .5:
                self.sigma *= self.gamma
            # update the penalty factor
            w = w - self.sigma * _eq_cons
            v = torch.max(v - self.sigma * _ieq_cons, torch.zeros_like(v))

        print('The optimal objective value is %f' % self.f(self.parameters).item())
        # save the result
        self.save_result()
        # is_plot
        #self.plot()
