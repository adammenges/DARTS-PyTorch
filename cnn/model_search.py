import  torch.nn.functional as F
from    operations import *
from    torch.autograd import Variable
from    genotypes import PRIMITIVES
from    genotypes import Genotype


class MixedOp(nn.Module):
    """
    a mixtures output of 8 type of units.
    we use weights to aggregate these outputs while training.
    and softmax to select the strongest edges while inference.
    """
    def __init__(self, C, stride):
        """

        :param C: 16
        :param stride: 1
        """
        super(MixedOp, self).__init__()

        self._ops = nn.ModuleList()
        """
        PRIMITIVES = [
                    'none',
                    'max_pool_3x3',
                    'avg_pool_3x3',
                    'skip_connect',
                    'sep_conv_3x3',
                    'sep_conv_5x5',
                    'dil_conv_3x3',
                    'dil_conv_5x5'
                ]
        """
        for primitive in PRIMITIVES:
            # create corresponding layer
            # [c] => [c], channels untouched.
            op = OPS[primitive](C, stride, False)
            # append batchnorm after pool layer
            if 'pool' in primitive:
                # disable affine w/b for batchnorm
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))

            self._ops.append(op)

    def forward(self, x, weights):
        """

        :param x: data
        :param weights: alpha, the output = sum of alpha * op(x)
        :return:
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))






class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        """

        :param steps: 4
        :param multiplier: 4
        :param C_prev_prev: 48
        :param C_prev: 48
        :param C: 16
        :param reduction: False
        :param reduction_prev: False
        """
        super(Cell, self).__init__()


        self.reduction = reduction

        # preprocess0 deal with output from prev_prev cell
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps # 4
        self._multiplier = multiplier # 4

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                # every intermediate node will connect its input and previous cell output and
                # prev_prev cell output
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        """

        :param s0:
        :param s1:
        :param weights:
        :return:
        """
        s0 = self.preprocess0(s0) # [40, 48, 32, 32], [40, 16, 32, 32]
        s1 = self.preprocess1(s1) # [40, 48, 32, 32], [40, 16, 32, 32]

        states = [s0, s1]
        offset = 0
        # for each node, receive input from all previous intermediate nodes and s0, s1
        for i in range(self._steps): # 4
            # [40, 16, 32, 32]
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1) # 6 of [40, 16, 32, 32]






class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        """

        :param C: 16
        :param num_classes: 10
        :param layers: 8
        :param criterion:
        :param steps: 4
        :param multiplier: 4
        :param stem_multiplier: 3
        """
        super(Network, self).__init__()

        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C # 48
        self.stem = nn.Sequential( # 3 => 48
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C # 48, 48, 16
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        """

        :param input:
        :return:
        """
        s0 = s1 = self.stem(input) # [b, 3, 32, 32], [b, 48, 32, 32]
        for i, cell in enumerate(self.cells):
            # weights are shared across all reduction cell or normal cell
            if cell.reduction: # if current cell is reduction cell
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1) # [14, 8]
            s0, s1 = s1, cell(s0, s1, weights) # [40, 64, 32, 32]

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits

    def _loss(self, input, target):
        """

        :param input:
        :param target:
        :return:
        """
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        """

        :return:
        """
        # k is the total number of edges inside single cell
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters




    def genotype(self):
        """

        :return:
        """
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none'))
                               )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
