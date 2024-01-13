from utils.model_summary import get_model_flops, get_model_activation
from model.Restormer import Restormer

model = Restormer()

input_dim = (3, 256, 256)  # set the input dimension
activations, num_conv = get_model_activation(model, input_dim)
activations = activations / 10 ** 6
print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

flops = get_model_flops(model, input_dim, False)
flops = flops / 10 ** 9
print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
num_parameters = num_parameters / 10 ** 6
print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))